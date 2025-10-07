# rl_eval.py
# -----------------------------------------------------------------------------
# Evaluate a trained RL policy (PPO or SAC) on a Reacher-based environment.
# Supports optional Reward Machine wrapper (--rm) for multi-goal evaluation.
# Computes control energy (Σ dt·||u||²) and accuracy metrics, optionally per-RM goal.
# -----------------------------------------------------------------------------

import os, os.path, time, argparse, json
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from reacher_obstacles.envs.reacher_v6 import CONFIGS
from reacher_obstacles.utils.experiments import EXPERIMENTS

# ---- RM imports (optional, used only if --rm) ----
try:
    from reacher_obstacles.rm.reward_machine import load_rm_spec
    from reacher_obstacles.rm.labeller import Labeller
    from reacher_obstacles.wrappers.rm_wrapper import RMWrapper
    _RM_AVAILABLE = True
except Exception:
    _RM_AVAILABLE = False

# optional YAML loader (for --rm-waypoints)
try:
    import yaml  # type: ignore
    _YAML_AVAILABLE = True
except Exception:
    _YAML_AVAILABLE = False

# ensure output folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("log", exist_ok=True)
os.makedirs("images", exist_ok=True)

# -----------------------------------------------------------------------------
# CLI arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, default="SAC", help="Algorithm")
assert(parser.parse_known_args()[0].algo in ["SAC", "PPO"])
model_class = SAC if parser.parse_known_args()[0].algo == "SAC" else PPO

parser.add_argument("expid", type=str, help="Experiment id (e.g., 1a)")
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--sjrs", action="store_true", help="Load ;SJRS model")
parser.add_argument("--model-name", type=str, default='', help="Model name for plot title")
parser.add_argument("--suffix", type=str, default=None, help="Exact suffix override (e.g., ';SJRS')")

# ---------- Reward Machine (optional; OFF unless --rm) ----------
parser.add_argument("--rm", action="store_true", help="Use Reward Machine wrapper for eval")
parser.add_argument("--rm-spec", type=str, default=None, help="Path to RM spec (yaml/json)")
parser.add_argument("--rm-eps", type=float, default=0.03, help="Labeller radius for near_Gi")
parser.add_argument("--rm-waypoints", type=str, default=None,
                    help="Optional JSON/YAML {name:[x,y,z]}; defaults to env CONFIG target as G1")
parser.add_argument("--rm-route-target", action="store_true",
                    help="Route env target when RM state changes (visual only)")
parser.add_argument("--rm-map", type=str, default=None,
                    help='Mapping from RM state to waypoint name, e.g. {"u1":"G1","u2":"G2"}')
parser.add_argument("--rm-reward-mode", choices=["add", "replace"], default="add")

# ---------- Viewer cosmetics ----------
parser.add_argument("--hide-builtin-target", action="store_true",
                    help="Hide env’s red target marker in viewer (default on)",
                    default=True)

args = parser.parse_args()

# -----------------------------------------------------------------------------
# Resolve experiment → model paths
# -----------------------------------------------------------------------------
expid = args.expid
seed = args.seed
envid = EXPERIMENTS[expid]['envid']

# suffix selection (priority: explicit -> sjrs flag -> none)
if args.suffix is not None:
    suffix = args.suffix
else:
    suffix = ";SJRS" if args.sjrs else ""

model_file = f"models/{envid};{seed};{args.algo}{suffix}"
config_key = envid.split('_')[1]
target_pos = np.array([*CONFIGS[config_key]['target'], 0.015])  # baseline target (for accuracy metric)

# -----------------------------------------------------------------------------
# Viewer helpers
# -----------------------------------------------------------------------------
def _draw_waypoints(env, wps: dict):
    """Draw labeled waypoint markers each render frame."""
    if not wps:
        return
    try:
        env.render()
        viewer = env.unwrapped.mujoco_renderer.viewer
    except Exception:
        return
    colors = [
        [0.9, 0.2, 0.2, 1.0], [0.2, 0.6, 0.9, 1.0],
        [0.2, 0.8, 0.3, 1.0], [0.9, 0.7, 0.2, 1.0],
        [0.6, 0.2, 0.7, 1.0], [0.2, 0.7, 0.7, 1.0],
    ]
    for k, (name, pos) in enumerate(wps.items()):
        p = np.asarray(pos, float).reshape(3)
        viewer.add_marker(pos=p, size=0.02, label=name, rgba=colors[k % len(colors)], type=2)


def _hide_builtin_target(env):
    """
    Make the original target site/geom invisible without altering obs.
    Tries in order: site 'target' → geom 'target' → geoms attached to body 'target'.
    """
    un = env.unwrapped
    m, s = getattr(un, "model", None), getattr(un, "sim", None)
    if m is None:
        return False
    hidden = False
    # Try successive options; silence missing handles
    try:
        sid = m.site("target").id
        if hasattr(m, "site_rgba"):
            rgba = m.site_rgba.copy()
            rgba[sid, 3] = 0.0
            m.site_rgba[:] = rgba
            hidden = True
    except Exception:
        pass
    if not hidden:
        try:
            gid = m.geom("target").id
            if hasattr(m, "geom_rgba"):
                rgba = m.geom_rgba.copy()
                rgba[gid, 3] = 0.0
                m.geom_rgba[:] = rgba
                hidden = True
        except Exception:
            pass
    if not hidden:
        try:
            bid = m.body("target").id
            if hasattr(m, "geom_rgba") and hasattr(m, "geom_bodyid"):
                for gid in range(m.ngeom):
                    if int(m.geom_bodyid[gid]) == int(bid):
                        m.geom_rgba[gid, 3] = 0.0
                hidden = True
        except Exception:
            pass
    try:
        if hidden and s is not None:
            s.forward()  # refresh viewer
    except Exception:
        pass
    return hidden

# -----------------------------------------------------------------------------
# Load optional waypoint/mapping files for RM
# -----------------------------------------------------------------------------
def _load_waypoints(path: str):
    if path is None:
        return None
    with open(path, "r") as f:
        if path.lower().endswith((".yml", ".yaml")):
            if not _YAML_AVAILABLE:
                raise RuntimeError("PyYAML not installed; provide JSON instead.")
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    return {k: np.asarray(v, float).reshape(3) for k, v in data.items()}

def _load_mapping(path: str):
    if path is None:
        return {}
    try:
        import yaml; _yaml = True
    except Exception:
        _yaml = False
    with open(path, "r") as f:
        data = yaml.safe_load(f) if _yaml and path.lower().endswith((".yml",".yaml")) else json.load(f)
    # convert keys/values to strings, None for "null"
    return {str(k): (None if v in [None, "null"] else str(v)) for k, v in data.items()}

# -----------------------------------------------------------------------------
# RM wrapper integration (only active if --rm)
# -----------------------------------------------------------------------------
def _wrap_with_rm(env: gym.Env, envid: str):
    if not args.rm:
        return env
    if not _RM_AVAILABLE:
        raise RuntimeError("Reward Machine modules not found.")
    if not args.rm_spec:
        raise ValueError("--rm requires --rm-spec")

    rm = load_rm_spec(args.rm_spec)
    wps = _load_waypoints(args.rm_waypoints)

    # fallback: single waypoint G1 = default target from CONFIGS
    if wps is None:
        key = envid.split("_")[1] if "_" in envid else None
        if key and key in CONFIGS and "target" in CONFIGS[key]:
            wps = {"G1": np.array([*CONFIGS[key]["target"], 0.015], dtype=float)}
        else:
            print("[RM] Warning: no waypoints; near_Gi transitions may never trigger.")
            wps = {}

    labeller = Labeller(wps, global_eps=float(args.rm_eps))
    u_to_wp = _load_mapping(args.rm_map)

    env = RMWrapper(
        env,
        rm=rm,
        labeller=labeller,
        w_rm=0.0,  # keep neutral at eval (RM reward not used)
        route_target=bool(args.rm_route_target),
        waypoint_order=u_to_wp,
        reward_mode=args.rm_reward_mode,
    )
    return env

def make_eval_env(render_mode=None):
    env = gym.make(envid, render_mode=render_mode) if render_mode else gym.make(envid)
    if args.rm:
        env = _wrap_with_rm(env, envid)
    return env

# -----------------------------------------------------------------------------
# Build evaluation environment + load model
# -----------------------------------------------------------------------------
eval_env = make_eval_env()
print(f"Observation (eval env): {eval_env.observation_space}")
print(f"Action (eval env):       {eval_env.action_space}")
print(f"MODEL FILE:               {model_file}.pth")

# load model file and attach matching env (ensures obs/action shape consistency)
if os.path.isfile(model_file + ".pth"):
    model = model_class.load(model_file + ".pth", eval_env)
    # reload replay buffer if present (for off-policy)
    if issubclass(model_class, OffPolicyAlgorithm):
        rb_path = model_file + "_rb.pth"
        if os.path.isfile(rb_path):
            try:
                model.load_replay_buffer(rb_path)
            except Exception:
                pass
    print(f"Loaded timesteps: {model.num_timesteps}")
else:
    raise FileNotFoundError(f"Model not found: {model_file}.pth")

# -----------------------------------------------------------------------------
# Rollout loop
# -----------------------------------------------------------------------------
render_mode = "human"
env = make_eval_env(render_mode=render_mode)
dt_attr = getattr(env.unwrapped, "dt", None)
dt = float(dt_attr if dt_attr is not None else 0.01)
print(f"dt (s): {dt:.6f}")

U, errors = [], []
obs, info = env.reset(seed=seed)

wps = _load_waypoints(args.rm_waypoints) or {}
u_to_wp = _load_mapping(args.rm_map) if args.rm_map else {}

errors_cfg = []  # distance to original config target
errors_rm  = []  # distance to current RM waypoint
_hidden_builtin = False

for t in range(300):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # log routing/goal updates if RM+RH combined
    if info.get("routed_to") or info.get("rh_rebuilt_goal"):
        print(f"[DBG] RM→ {info.get('wp_name')}  RMpos={info.get('wp_pos')}  RHgoal={info.get('rh_goal')}")

    # store action for energy computation
    u = np.asarray(action, dtype=np.float64).copy()
    U.append(u)

    # --- accuracy errors ---
    ee = env.unwrapped.data.body("fingertip").xpos[0:env.unwrapped.ndim]
    ee = np.array([*ee, 0.015])
    errors_cfg.append(np.linalg.norm(ee - target_pos))  # vs base target

    # error vs RM subgoal (if mapped)
    rm_u = info.get("rm_state")
    wp_name = u_to_wp.get(rm_u)
    target_rm = wps.get(wp_name, target_pos)
    errors_rm.append(np.linalg.norm(ee - np.asarray(target_rm, float)))

    # --- rendering / visualization ---
    if render_mode == "human":
        if args.rm:
            if args.hide_builtin_target and not _hidden_builtin:
                _hide_builtin_target(env)
                _hidden_builtin = True
            _draw_waypoints(env, wps)
        # yellow trail for fingertip
        env.unwrapped.mujoco_renderer.viewer.add_marker(
            pos=ee, size=0.005, label="", rgba=[1, 1, 0, 1], type=2
        )
        env.render()
        time.sleep(0.05)

    if terminated or truncated:
        print(info)
        break

# -----------------------------------------------------------------------------
# Post-evaluation metrics
# -----------------------------------------------------------------------------
U = np.vstack(U) if len(U) else np.zeros((0, 0))
T = U.shape[0]
E_action_tot = float(np.sum(np.sum(U**2, axis=1)) * dt) if T > 0 else float("nan")
E_action_mean = E_action_tot / max(T, 1)

print(f"shapes U: {U.shape}")
print(f"PAPER ENERGY (Σ dt·||u||²): {E_action_tot:.6f}  (mean: {E_action_mean:.6f})")
if not args.rm:
    print(f"ACC ERROR: {np.mean(errors) if errors else float('nan')}")
else:
    print(f"ACC ERROR (vs RM waypoint): {np.mean(errors_rm) if errors_rm else float('nan')}")

# -----------------------------------------------------------------------------
# Plot torque/action profiles
# -----------------------------------------------------------------------------
plt.figure()
plt.plot(U)
plt.title(f"{args.model_name} | Mean Error={np.mean(errors) if errors else np.mean(errors_rm):.3f} "
          f" Energy={E_action_tot:.3f}")
plt.xlabel("Time step")
plt.ylabel("u ([-1,1])")
if U.size:
    plt.ylim(-1, 1)
plt.legend([f"Joint {i+1}" for i in range(U.shape[1])])
plt.savefig(f"images/{expid}_actions.png", dpi=300, bbox_inches="tight")

env.close()
eval_env.close()
