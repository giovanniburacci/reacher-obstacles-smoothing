import os, os.path, argparse
import json

import gymnasium as gym
import numpy as np

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, BaseCallback

import reacher_obstacles.envs
from reacher_obstacles.envs.reacher_v6 import CONFIGS
from reacher_obstacles.utils.experiments import EXPERIMENTS
from reacher_obstacles.utils.sjrs import SJRSRewardWrapper, SJRSCallback

# --- Optional: RM modules (only used if --rm is passed)
try:
    from reacher_obstacles.rm.reward_machine import load_rm_spec
    from reacher_obstacles.rm.labeller import Labeller
    from reacher_obstacles.wrappers.rm_wrapper import RMWrapper
    _RM_AVAILABLE = True
except Exception:
    _RM_AVAILABLE = False

# optional YAML support for waypoints file
try:
    import yaml  # type: ignore
    _YAML_AVAILABLE = True
except Exception:
    _YAML_AVAILABLE = False

os.makedirs("models", exist_ok=True)
os.makedirs("log", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, default="PPO")
assert(parser.parse_known_args()[0].algo in ["SAC","PPO"])
model_class = SAC if parser.parse_known_args()[0].algo == "SAC" else PPO

parser.add_argument("expid", type=str, help="Experiment id (e.g. 4b)")
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--steps", type=int, default=None, help="Override train steps")

# SJRS (reward wrapper)
parser.add_argument("--sjrs", action="store_true")
parser.add_argument("--sjrs-lambda-t", type=float, default=0.0)
parser.add_argument("--sjrs-lambda-a", type=float, default=0.0)
parser.add_argument("--sjrs-lambda-c", type=float, default=0.0)
parser.add_argument("--sjrs-mode",
                    choices=["torque", "action"],
                    default="torque",
                    help="Signal SJRS penalizes: physical torque (tau) or raw action (u).")

# ----- Reward Machine (optional; OFF unless --rm is passed) -----
parser.add_argument("--rm", action="store_true", help="Enable Reward Machine wrapper (cross-product baseline)")
parser.add_argument("--rm-spec", type=str, default=None, help="Path to RM spec (yaml/json). Required if --rm")
parser.add_argument("--rm-weight", type=float, default=1.0, help="Weight for sparse RM progress reward")
parser.add_argument("--rm-eps", type=float, default=0.03, help="Labeller radius for near_Gi")
parser.add_argument("--rm-waypoints", type=str, default=None,
                    help="(Optional) JSON/YAML mapping {name: [x,y,z]} for labeller. "
                         "If omitted, defaults to single waypoint G1 from CONFIGS[key]['target'].")
parser.add_argument("--rm-route-target", action="store_true",
                    help="Try to route env target by current RM state -> waypoint (default: off)")
parser.add_argument("--rm-map", type=str, default=None,
                    help='JSON/YAML mapping from RM state to waypoint name, e.g. {"u1":"G1","u2":"G2","uF":"G3"}')
parser.add_argument("--rm-reward-mode", choices=["add", "replace"], default="add")

# vec env
parser.add_argument("--n-envs", type=int, default=8)

parser.add_argument("--n-steps", type=int, default=2048)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--learning-rate", type=float, default=3e-4)

parser.add_argument("--tb", action="store_true", help="Enable TensorBoard logging")

args = parser.parse_args()
from stable_baselines3.common.callbacks import BaseCallback

class RMLoggingCallback(BaseCallback):
    """
    Aggregates RM/RH diagnostics from env infos and logs per-episode to TensorBoard.
    Works with VecEnv (handles multiple envs).
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.buf = {}  # per-env accumulators

    def _init_buf(self, i):
        self.buf[i] = dict(
            r_env=0.0, r_rm=0.0, r_tot=0.0, steps=0,
            hits=0, routed=0, trans=0,
            first_u1=None, first_u2=None, first_uF=None,
            dist_sum=0.0, dist_cnt=0
        )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for i, info in enumerate(infos):
            if i not in self.buf:
                self._init_buf(i)
            b = self.buf[i]
            if not info:
                continue
            # sums
            b["r_env"] += float(info.get("env_reward", 0.0))
            b["r_rm"]  += float(info.get("rm_reward", 0.0))
            b["r_tot"] += float(info.get("total_reward", 0.0))
            b["steps"] += 1
            # events
            if "hit" in info.get("sigma", []):
                b["hits"] += 1
            if "routed_to" in info:
                b["routed"] += 1
            # transitions (rm_state change is visible via routed_to; also derive from rm_state)
            u = info.get("rm_state")
            t = self.num_timesteps
            if u == "u1" and b["first_u1"] is None:
                b["first_u1"] = t
            if u == "u2" and b["first_u2"] is None:
                b["first_u2"] = t
            if u == "uF" and b["first_uF"] is None:
                b["first_uF"] = t
            # distance to current waypoint
            d = info.get("dist_to_wp")
            if d is not None and np.isfinite(d):
                b["dist_sum"] += float(d); b["dist_cnt"] += 1

            # episode end?
            if isinstance(dones, (list, tuple, np.ndarray)) and i < len(dones) and dones[i]:
                # log to TB
                steps = max(1, b["steps"])
                self.logger.record("rm/episode_r_env", b["r_env"])
                self.logger.record("rm/episode_r_rm",  b["r_rm"])
                self.logger.record("rm/episode_r_total", b["r_tot"])
                self.logger.record("rm/episode_hits",   b["hits"])
                self.logger.record("rm/episode_routed", b["routed"])
                self.logger.record("rm/mean_dist_to_wp", b["dist_sum"]/max(1,b["dist_cnt"]))
                if b["first_u1"] is not None: self.logger.record("rm/first_hit_u1_t", float(b["first_u1"]))
                if b["first_u2"] is not None: self.logger.record("rm/first_hit_u2_t", float(b["first_u2"]))
                if b["first_uF"] is not None: self.logger.record("rm/first_hit_uF_t", float(b["first_uF"]))
                self.logger.dump(self.num_timesteps)
                # reset buffer
                self._init_buf(i)
        return True

# =======================
# RM helpers (used only if --rm)
# =======================
def _load_waypoints(path: str):
    """Load a dict {name: [x,y,z]} from JSON or YAML."""
    if path is None:
        return None
    with open(path, "r") as f:
        if path.lower().endswith((".yml", ".yaml")):
            if not _YAML_AVAILABLE:
                raise RuntimeError("PyYAML not installed; provide JSON or install pyyaml.")
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    # normalize to np.array float
    wps = {k: np.asarray(v, dtype=float).reshape(3) for k, v in data.items()}
    return wps

def _load_mapping(path: str):
    if path is None:
        return {}
    import json
    try:
        import yaml
        _yaml = True
    except Exception:
        _yaml = False
    with open(path, "r") as f:
        data = yaml.safe_load(f) if _yaml and path.lower().endswith((".yml",".yaml")) else json.load(f)
    # normalize nulls/'null'
    return {str(k): (None if v in [None, "null"] else str(v)) for k, v in data.items()}


def _envid_to_key(envid: str):
    # expects something like "Reacher3-v6_FTO3_rhV"
    return envid.split("_")[1] if "_" in envid else None

def _wrap_with_rm(env: gym.Env, envid: str, args):
    if not _RM_AVAILABLE:
        raise RuntimeError("Reward Machine modules not available. Ensure reacher_obstacles.rm.* files exist.")
    if not args.rm_spec:
        raise ValueError("--rm requires --rm-spec")

    rm = load_rm_spec(args.rm_spec)

    # waypoints for labeller: from file if provided, else fallback to CONFIGS target as single G1
    wps = _load_waypoints(args.rm_waypoints)
    if wps is None:
        key = _envid_to_key(envid)
        if key and key in CONFIGS and "target" in CONFIGS[key]:
            # append fixed z like elsewhere in repo (0.015)
            wps = {"G1": np.array([*CONFIGS[key]["target"], 0.015], dtype=float)}
        else:
            # no waypoints → RM transitions based on near_Gi won't fire; warn early
            print("[RM] Warning: No waypoints provided and no CONFIGS[key]['target'] found. "
                  "RM progress rewards may never trigger.")

    labeller = Labeller(wps or {}, global_eps=float(args.rm_eps))

    u_to_wp = _load_mapping(args.rm_map)  # e.g., {"u1":"G1","u2":"G2","uF":"G3"}
    env = RMWrapper(
        env,
        rm=rm,
        labeller=labeller,
        w_rm=float(args.rm_weight),
        route_target=bool(args.rm_route_target),
        waypoint_order=u_to_wp,
        reward_mode=args.rm_reward_mode,  # <— strict baseline: RM reward only
    )

    return env


if __name__ == "__main__":
    expid = args.expid
    seed = args.seed

    envid = EXPERIMENTS[expid]['envid']
    train_steps = int(EXPERIMENTS[expid]['train_steps'])
    if args.steps is not None:
        train_steps = int(args.steps)

    # make sure TB base dir exists if enabled
    if args.tb:
        os.makedirs("log/tb", exist_ok=True)

    # ----- env factory with wrappers -----
    def make_one_env():
        env = gym.make(envid)

        # RM wrapping
        if args.rm:
            env = _wrap_with_rm(env, envid, args)

        # SJRS wrapping
        if args.sjrs and (args.sjrs_lambda_t > 0 or args.sjrs_lambda_a > 0 or args.sjrs_lambda_c > 0):
            env = SJRSRewardWrapper(
                env,
                lambda_t=args.sjrs_lambda_t,
                lambda_a=args.sjrs_lambda_a,
                lambda_c=args.sjrs_lambda_c,
                sjrs_mode=args.sjrs_mode,
            )
        return env

    env_fns = [make_one_env for _ in range(args.n_envs)]
    base_env = SubprocVecEnv(env_fns)

    # wrap with VecMonitor for reliable episodic stats/logging
    train_env = VecMonitor(base_env)

    # ----- model (PPO defaults from SB3) -----
    if issubclass(model_class, OnPolicyAlgorithm):
        model = PPO(
            "MlpPolicy",
            train_env,
            seed=seed,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=0.99,
            gae_lambda=0.95,
            verbose=1,
            tensorboard_log=("log/tb" if args.tb else None)
        )
    else:
        model = SAC(
            "MlpPolicy",
            train_env,
            seed=seed,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            verbose=1,
        )

    # ----- callbacks -----
    callbacks = []
    if args.sjrs:
        # SJRS logging callback
        callbacks.append(SJRSCallback(verbose=0))
    if args.rm:
        callbacks.append(RMLoggingCallback())
    callback = CallbackList(callbacks) if callbacks else None

    # ----- train -----
    print(f"Training {envid};{seed} for {train_steps} timesteps; "
          f"{'SJRS ' if args.sjrs else ''}{'RM ' if args.rm else ''}")
    if args.sjrs:
        print(f"[SJRS] mode={args.sjrs_mode}")
    if args.rm:
        print(f"[RM] spec={args.rm_spec} | weight={args.rm_weight} | eps={args.rm_eps}"
              f"{' | route_target=ON' if args.rm_route_target else ''}"
              f"{f' | wps={args.rm_waypoints}' if args.rm_waypoints else ''}")

    # NOTE: suffix stays ONLY ;SJRS (no ;RM)
    run_name = f"{envid};{seed};{args.algo}"
    if args.sjrs:
        run_name += ";SJRS"

    model.learn(
        total_timesteps=train_steps,
        callback=callback,
        log_interval=100,
        tb_log_name=run_name if args.tb else None
    )

    # ----- save -----
    suffix = ""
    if args.sjrs:
        suffix += ";SJRS"
    model_path = f"models/{envid};{seed};{args.algo}{suffix}.pth"
    model.save(model_path)
    print("Saved:", model_path)

    train_env.close()
