import os, os.path, time, argparse
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from reacher_obstacles.envs.reacher_v6 import CONFIGS
from reacher_obstacles.utils.experiments import EXPERIMENTS

os.makedirs("models", exist_ok=True)
os.makedirs("log", exist_ok=True)
os.makedirs("images", exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, default="SAC", help="Algorithm")
assert(parser.parse_known_args()[0].algo in ["SAC", "PPO"])
model_class = SAC if parser.parse_known_args()[0].algo == "SAC" else PPO

parser.add_argument("expid", type=str, help="Experiment id")
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--sjrs", action="store_true", help="Load ;SJRS model")

parser.add_argument("--model-name", type=str, default='', help="Model name for torque plot")

# optional power-user override
parser.add_argument("--suffix", type=str, default=None, help="Exact suffix to append (e.g., ';SJRS')")

args = parser.parse_args()

expid = args.expid
seed  = args.seed

envid = EXPERIMENTS[expid]['envid']
suffix = ""
if args.suffix is not None:
    suffix = args.suffix
else:
    suffix = ""
    if args.sjrs:
        suffix += ";SJRS"

model_file = f"models/{envid};{seed};{args.algo}{suffix}"

config = envid.split('_')[1]
target_pos = np.array([*CONFIGS[config]['target'], 0.015])

train_env = gym.make(envid)
print(f"Observation: {train_env.observation_space}")
print(f"Action: {train_env.action_space}")
print(f"MODEL FILE: {model_file}.pth")

if os.path.isfile(model_file + ".pth"):
    model = model_class.load(model_file + ".pth", train_env)
    if issubclass(model_class, OffPolicyAlgorithm):
        rb_path = model_file + "_rb.pth"
        if os.path.isfile(rb_path):
            model.load_replay_buffer(rb_path)
    print(f"Loaded timesteps: {model.num_timesteps}")
else:
    raise FileNotFoundError(f"Model not found: {model_file}.pth")

render_mode = "human"
env = gym.make(envid, render_mode=render_mode)
dt = float(getattr(env.unwrapped, "dt"))
print(f"dt (s): {dt:.6f}")

# sizes
n_act = env.unwrapped.model.nu
n_dof = env.unwrapped.model.nv
assert n_act <= n_dof, "Assuming 1 actuator per joint subset"

trajectory, errors = [], []
U, TAU, QVEL, TAU_NORM = [], [], [], []  # <-- added TAU_NORM
obs, info = env.reset(seed=seed)

for t in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # logging
    u   = np.asarray(action, dtype=np.float64).copy()              # (n_act,)
    tau = np.asarray(env.unwrapped.data.qfrc_actuator, np.float64)[:n_act].copy()

    U.append(u)
    TAU.append(tau)
    # -----------------------------------------

    ee = env.unwrapped.data.body("fingertip").xpos[0:env.unwrapped.ndim]
    ee = np.array([*ee, 0.015])
    trajectory.append(ee)
    errors.append(np.linalg.norm(ee - target_pos))

    if t < 60 or (t % 10 == 0):
        print(f"u(ctrl): {u} | tau: {tau} | qacc: {env.unwrapped.data.qacc[:n_act]}")

    if render_mode == "human":
        for p in trajectory:
            env.unwrapped.mujoco_renderer.viewer.add_marker(
                pos=p, size=0.005, label="", rgba=[1,1,0,1], type=2)
        env.render()
        time.sleep(0.05)

    if terminated or truncated:
        print(info)
        break

U        = np.vstack(U)        # (T, n_act)
TAU      = np.vstack(TAU)      # (T, n_act)

# --------- metrics ----------
E_tau = float(np.mean(np.sum(TAU**2, axis=1)) * dt)          # physical effort per-step mean (dt*||tau||^2)


print(f"shapes U,TAU: {U.shape} {TAU.shape}")

# ---- Energy metrics (episode) ----
T = U.shape[0]
dt_used = float(locals().get("dt", getattr(env.unwrapped, "dt", 0.01)))

E_paper_tot = float(np.sum(np.sum(U**2, axis=1)) * dt_used)
E_paper_mean = E_paper_tot / max(T, 1)

print(f"PAPER ENERGY (Σ dt·||u||²):             {E_paper_tot:.6f}  (per-step mean: {E_paper_mean:.6f})")
print(f"ACC ERROR: {np.mean(errors)}")

# --------- plots ----------
plt.figure()
plt.plot(U)
plt.title(f"{args.model_name} | Mean Error={np.mean(errors):.3f}  Energy={E_paper_tot:.3f}")
plt.xlabel("Time step"); plt.ylabel("u ([-1,1])")
plt.ylim(-1, 1)
plt.legend([f"Joint {i+1}" for i in range(U.shape[1])])
plt.savefig(f"images/{expid}_torques.png", dpi=300, bbox_inches="tight")

env.close()
