import argparse
import os
import time
import casadi
import numpy as np

from reacher_obstacles.utils import project_root, src_dir
import mujoco
from reacher_obstacles.trajopt.robot_model import RobotModel
import matplotlib.pyplot as plt

from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from reacher_obstacles.utils.experiments import EXPERIMENTS
from reacher_obstacles.envs.reacher_v6 import CONFIGS

from reacher_obstacles.trajopt.reacher_trajopt_seq import ReacherTrajoptSequential  # sequential solver

xml_path = f"{src_dir()}/envs/assets/reacher3.xml"
parser = argparse.ArgumentParser()

parser.add_argument("expid", type=str, default="1a", help="Experiment id")
parser.add_argument("--nsteps", type=int, default=100, help="Number of steps")
parser.add_argument("--force-training", action="store_true", help="Force training")

args = parser.parse_args()
expid = args.expid
nsteps = args.nsteps
envid: str = EXPERIMENTS[expid]['envid']
config: str = envid.split('_')[1]

# Obstacles from CONFIG
try:
    obstacles_pos = CONFIGS[config]['obstacles']
except KeyError:
    obstacles_pos = []

print(f"Experiment: {expid}")
print(f"Obstacles position: {obstacles_pos}")

traj_path = f"{project_root()}/trajectories/trajectory_{expid}.npz"
if (not os.path.isfile(traj_path)) or args.force_training:
    print("Generating trajectory with ReacherTrajoptSequential...")
    qpos = np.array([-0.5, 0.0, 0.2])
    seq_opt = ReacherTrajoptSequential(xml_path, obstacles_pos, nsteps=nsteps, expid=expid)
    X, A, U = seq_opt.solve_sequence(targets, q0=qpos)
    phase_lengths = np.array([nsteps] * len(targets), dtype=int)  # save boundaries
    os.makedirs(f"{project_root()}/trajectories", exist_ok=True)
    np.savez(traj_path, X=X, A=A, U=U, targets=np.array(targets), phase_lengths=phase_lengths)

if os.path.isfile(traj_path):
    print("FOUND TRAJECTORY")
    T = np.load(traj_path, allow_pickle=True)
    X = T['X']; A = T['A']; U = T['U']
    # load targets & phase lengths if present; else fallback for old files
    targets = T['targets']
    phase_lengths = T['phase_lengths'] if 'phase_lengths' in T.files else np.array([len(U)], dtype=int)
else:
    raise FileNotFoundError("Trajectory not found")

# Build robot for simulation/visualization. Use last target to init model
final_target = np.array(targets[-1], dtype=float)
robot = RobotModel(xml_path, final_target, obstacles_pos)
mj_model = robot.mj_model
mj_data = robot.mj_data

# Simulate and display video with per-phase error tracking
render_mode = "human"
frames, trajectory, accelerations, torques, errors = [], [], [], [], []

renderer = MujocoRenderer(mj_model, mj_data)

phase_cum = np.cumsum(phase_lengths)
def active_target_idx(t):
    return int(np.searchsorted(phase_cum, t, side='right'))

for t, torque in enumerate(U):
    qacc, qvel, qpos = robot.apply_torque(torque)
    mj_data.qpos[:robot.nq] = robot.qpos
    mujoco.mj_step(mj_model, mj_data)

    ee_pos = mj_data.body("fingertip").xpos[0:2]
    ee_pos = np.array([*ee_pos, 0.015])
    trajectory.append(ee_pos)

    idx = active_target_idx(t)
    tgt = np.array(targets[idx], dtype=float)
    errors.append(np.linalg.norm(ee_pos - tgt))

    accelerations.append(qacc)
    torques.append(torque)

    if render_mode == "human":
        pixels = renderer.render("human")
        time.sleep(0.1)
        for p in trajectory:
            renderer.viewer.add_marker(pos=p, size=0.005, label="", rgba=[1, 1, 0, 1], type=2)
        frames.append(pixels)

accelerations = np.array(accelerations)
torques = np.array(torques)

# Plot torques
plt.figure()
plt.plot(U)
plt.title("Torques over time")
plt.xlabel("Time step", fontdict={'size': 20}); plt.xticks(fontsize=15)
plt.ylabel("Torque [Nm]", fontdict={'size': 20}); plt.yticks(fontsize=15)
plt.legend([f"Joint {i+1}" for i in range(U.shape[1])], fontsize=13, loc='upper right')
plt.savefig(f"images/{expid}_trajopt.png", dpi=1000, bbox_inches='tight')

# Per-phase error summary
errors = np.array(errors, dtype=float)

# Global summaries
acc_error = float(np.sum(errors) / len(errors))
print(f"ACC ERROR (overall mean): {acc_error:.6f}")
print(f"SUMMED TORQUES: {np.sum(np.sum(torques ** 2, axis=1)) * 0.01:.6f}")

renderer.close()
