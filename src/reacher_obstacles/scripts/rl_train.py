import os, os.path, argparse
import gymnasium as gym

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, BaseCallback

import reacher_obstacles.envs
from reacher_obstacles.envs.reacher_v6 import CONFIGS
from reacher_obstacles.utils.experiments import EXPERIMENTS

from reacher_obstacles.utils.sjrs import SJRSRewardWrapper, SJRSCallback

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
                    help="Signal SJRS penalizes: physical torque (tau), raw action (u), or u scaled like torque.")

# vec env
parser.add_argument("--n-envs", type=int, default=8)

parser.add_argument("--n-steps", type=int, default=2048)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--learning-rate", type=float, default=3e-4)

parser.add_argument("--tb", action="store_true", help="Enable TensorBoard logging")

args = parser.parse_args()

class TBHeartbeat(BaseCallback):
    def __init__(self, interval=500, verbose=0):
        super().__init__(verbose)
        self.interval = interval

    def _on_training_start(self) -> None:
        self.logger.record("heartbeat/alive", 1.0)
        self.logger.dump(0)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.interval == 0:
            self.logger.record("heartbeat/steps", float(self.num_timesteps))
            # force flush so TensorBoard picks it up right away
            self.logger.dump(self.num_timesteps)
        return True

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
        # apply SJRS
        if args.sjrs and (args.sjrs_lambda_t>0 or args.sjrs_lambda_a>0 or args.sjrs_lambda_c>0):
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
    callbacks.append(TBHeartbeat(interval=500))
    callback = CallbackList(callbacks) if callbacks else None

    # ----- train -----
    print(f"Training {envid};{seed} for {train_steps} timesteps; "
          f"{'SJRS ' if args.sjrs else ''}")
    if args.sjrs:
        print(f"[SJRS] mode={args.sjrs_mode}")

    run_name = f"{envid};{seed};{args.algo}"
    if args.sjrs: run_name += ";SJRS"

    model.learn(
        total_timesteps=train_steps,
        callback=callback,
        log_interval=100,
        tb_log_name=run_name if args.tb else None
    )

    # ----- save -----
    suffix = ""
    if args.sjrs: suffix += ";SJRS"
    model_path = f"models/{envid};{seed};{args.algo}{suffix}.pth"
    model.save(model_path)
    print("Saved:", model_path)

    train_env.close()
