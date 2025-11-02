# trajopt_train.py — unified + RM-style waypoint markers in sequential mode

import argparse
import os
import time
import numpy as np
import yaml

from reacher_obstacles.utils import project_root, src_dir
import mujoco
from reacher_obstacles.trajopt.robot_model import RobotModel
from reacher_obstacles.trajopt.reacher_trajopt import ReacherTrajopt
from reacher_obstacles.trajopt.reacher_trajopt_seq import ReacherTrajoptSequential  # for sequential mode

from reacher_obstacles.utils.experiments import EXPERIMENTS
from reacher_obstacles.envs.reacher_v6 import CONFIGS

import matplotlib.pyplot as plt
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

xml_path = f"{src_dir()}/envs/assets/reacher3.xml"

# ---------- Viewer helpers (same idea as in trajopt_eval / rl_eval) ----------
def _hide_builtin_target_model(m):
    """Hide the original env target geom (set alpha=0)."""
    try:
        gid = m.geom("target").id
    except Exception:
        return False
    if hasattr(m, "geom_rgba"):
        rgba = m.geom_rgba.copy()
        rgba[gid, 3] = 0.0
        m.geom_rgba[:] = rgba
        return True
    return False

def _draw_waypoints_on_viewer(viewer, wps_dict):
    """Draw labeled waypoint markers (G1,G2,...) once."""
    if viewer is None or not wps_dict:
        return
    colors = [
        [0.9, 0.2, 0.2, 1.0], [0.2, 0.6, 0.9, 1.0],
        [0.2, 0.8, 0.3, 1.0], [0.9, 0.7, 0.2, 1.0],
        [0.6, 0.2, 0.7, 1.0], [0.2, 0.7, 0.7, 1.0],
    ]
    for k, (name, pos) in enumerate(wps_dict.items()):
        p = np.asarray(pos, float).reshape(3)
        viewer.add_marker(pos=p, size=0.02, label=name, rgba=colors[k % len(colors)], type=2)

# ---------- YAML loader (accepts dict or list; preserves names/order) ----------
def load_targets_from_yaml(path: str):
    """
    Returns:
      targets_list: [np.array([x,y,z]), ...] in file order
      names_list:   ["G1","G2",...] from YAML keys if dict, else auto G1..Gn
      wps_dict:     mapping name -> pos (for drawing)
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    targets_list, names_list = [], []
    if isinstance(data, dict):
        # dict with ordered keys (PyYAML preserves insertion order)
        for name, v in data.items():
            arr = np.array(v, dtype=float).reshape(-1)
            if arr.size == 2:
                arr = np.array([arr[0], arr[1], 0.015], dtype=float)
            else:
                arr = np.array([arr[0], arr[1], arr[2]], dtype=float)
            targets_list.append(arr)
            names_list.append(str(name))
    elif isinstance(data, list):
        for i, v in enumerate(data, start=1):
            arr = np.array(v, dtype=float).reshape(-1)
            if arr.size == 2:
                arr = np.array([arr[0], arr[1], 0.015], dtype=float)
            else:
                arr = np.array([arr[0], arr[1], arr[2]], dtype=float)
            targets_list.append(arr)
        names_list = [f"G{i}" for i in range(1, len(targets_list) + 1)]
    else:
        raise ValueError(f"No targets list found in YAML {path}. Provide dict or list.")

    wps_dict = {nm: pos for nm, pos in zip(names_list, targets_list)}
    return targets_list, names_list, wps_dict

parser = argparse.ArgumentParser()
parser.add_argument("expid", type=str, default="1a", help="Experiment id")
parser.add_argument("--nsteps", type=int, default=100, help="Number of steps (per phase in sequential mode)")
parser.add_argument("--targets-seq", type=str, default=None,
                    help="Path to YAML with sequential targets. If set, runs sequential trajopt.")
args = parser.parse_args()

expid = args.expid
nsteps = args.nsteps
targets_yaml = args.targets_seq

# Derive env + obstacles from experiment id
envid: str = EXPERIMENTS[expid]['envid']
config: str = envid.split('_')[1]

try:
    obstacles_pos = CONFIGS[config]['obstacles']
except KeyError:
    obstacles_pos = []

print(f"Experiment: {expid}")
print(f"Obstacles position: {obstacles_pos}")

_hidden_builtin = False

# Initial state
qpos = np.array([-0.5, 0.0, 0.2])

# ==== MODE SELECTION ====
sequential_mode = targets_yaml is not None

if sequential_mode:
    # ---------- SEQUENTIAL MODE ----------
    targets, names, wps_dict = load_targets_from_yaml(targets_yaml)
    print(f"Sequential targets loaded from {targets_yaml}: {targets}")

    # Solve sequentially
    seq_opt = ReacherTrajoptSequential(xml_path, obstacles_pos, nsteps=nsteps, expid=expid)
    X, A, U = seq_opt.solve_sequence(targets, q0=qpos)

    # Phase boundaries (constant horizon per phase)
    phase_lengths = np.array([nsteps] * len(targets), dtype=int)

    # Save with targets + phase_lengths for later eval
    os.makedirs(f"{project_root()}/trajectories", exist_ok=True)
    np.savez(
        f"{project_root()}/trajectories/trajectory_{expid}.npz",
        X=X, A=A, U=U,
        targets=np.array(targets),
        phase_lengths=phase_lengths,
        target_names=np.array(names),
    )

    # Reload
    traj_path = f"{project_root()}/trajectories/trajectory_{expid}.npz"
    if os.path.isfile(traj_path):
        print("FOUND TRAJECTORY")
        T = np.load(traj_path, allow_pickle=True)
        X = T['X']; A = T['A']; U = T['U']
        targets = T['targets']; phase_lengths = T['phase_lengths']
    else:
        raise FileNotFoundError("Trajectory not found")

    # Build robot for sim/vis; last target used to init model
    final_target = np.array(targets[-1], dtype=float)
    robot = RobotModel(xml_path, final_target, obstacles_pos, q0=qpos)

else:
    # ---------- SINGLE-TARGET MODE ----------
    target_pos = CONFIGS[config]['target']
    target_pos = np.array([*target_pos, 0.015])
    print(f"Target position: {target_pos}")

    robot = RobotModel(xml_path, target_pos, obstacles_pos, q0=qpos)
    reacher_task = ReacherTrajopt(robot, nsteps, expid)
    X, A, U = reacher_task.solve(robot.qpos)

    os.makedirs(f"{project_root()}/trajectories", exist_ok=True)
    np.savez(f"{project_root()}/trajectories/trajectory_{expid}.npz", X=X, A=A, U=U)

    traj_path = f"{project_root()}/trajectories/trajectory_{expid}.npz"
    if os.path.isfile(traj_path):
        print("FOUND TRAJECTORY")
        T = np.load(traj_path)
        X = T['X']; A = T['A']; U = T['U']
    else:
        raise FileNotFoundError("Trajectory not found")

    # For sim/vis
    final_target = target_pos
    wps_dict = {}  # no RM waypoints in single-target mode

# Simulation + plotting
mj_model = robot.mj_model
mj_data = robot.mj_data

render_mode = "human"
frames = []
trajectory = []
accelerations = []
torques = []
errors = []

renderer = MujocoRenderer(mj_model, mj_data)

# --- Initialize viewer once
if render_mode == "human":
    _ = renderer.render("human")  # ensure renderer.viewer is created
    if sequential_mode:
        _hide_builtin_target_model(mj_model)
        _draw_waypoints_on_viewer(renderer.viewer, wps_dict)

if sequential_mode:
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

    if sequential_mode:
        tgt = np.array(targets[active_target_idx(t)], dtype=float)
    else:
        tgt = final_target
    errors.append(np.linalg.norm(ee_pos - tgt))

    accelerations.append(qacc)
    torques.append(torque)

    if render_mode == "human":
        # Hide built-in target only in multi-target case, and draw RM-style markers
        if sequential_mode and not _hidden_builtin:
            _hidden_builtin = _hide_builtin_target_model(mj_model)
        if sequential_mode and wps_dict:
            _draw_waypoints_on_viewer(renderer.viewer, wps_dict)

        pixels = renderer.render("human")
        time.sleep(0.05)
        for p in trajectory:
            renderer.viewer.add_marker(pos=p, size=0.005, label = "", rgba=[1, 1, 0, 1], type=2)
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

# Error summaries
errors = np.array(errors, dtype=float)

if sequential_mode:
    start = 0
    for i, L in enumerate(phase_lengths):
        seg = errors[start:start+L]
        print(f"[Phase {i}] mean_error={seg.mean():.6f}  final_error={seg[-1]:.6f}  steps={L}")
        start += L

acc_error = float(np.sum(errors) / len(errors))
print(f"ACC ERROR (overall mean): {acc_error:.6f}")
print(f"SUMMED TORQUES: {np.sum(np.sum(torques ** 2, axis=1)) * 0.01:.6f}")

renderer.close()
