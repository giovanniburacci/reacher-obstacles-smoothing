import argparse
import os
import time
import numpy as np
import mujoco
import yaml
import matplotlib.pyplot as plt

from reacher_obstacles.utils import project_root, src_dir
from reacher_obstacles.trajopt.robot_model import RobotModel
from reacher_obstacles.utils.experiments import EXPERIMENTS
from reacher_obstacles.envs.reacher_v6 import CONFIGS
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

from reacher_obstacles.trajopt.reacher_trajopt_seq import ReacherTrajoptSequential

xml_path = f"{src_dir()}/envs/assets/reacher3.xml"

# ---------- Viewer helpers (mirroring rl_eval style) ----------
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
    if viewer is None:
        return
    if not wps_dict:
        return
    colors = [
        [0.9, 0.2, 0.2, 1.0], [0.2, 0.6, 0.9, 1.0],
        [0.2, 0.8, 0.3, 1.0], [0.9, 0.7, 0.2, 1.0],
        [0.6, 0.2, 0.7, 1.0], [0.2, 0.7, 0.7, 1.0],
    ]
    for k, (name, pos) in enumerate(wps_dict.items()):
        p = np.asarray(pos, float).reshape(3)
        viewer.add_marker(pos=p, size=0.02, label=name, rgba=colors[k % len(colors)], type=2)

# ---------- YAML loader that preserves names/order ----------
def load_targets_from_yaml(path: str):
    """
    Returns:
      targets_list: [np.array([x,y,z]), ...] in file order
      names_list:   ["G1","G2",...] from YAML keys if dict, else auto G1..Gn
      wps_dict:     Ordered dict-like mapping name -> pos (for drawing)
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    targets_list, names_list = [], []
    if isinstance(data, dict):
        for name, v in data.items():  # order preserved by PyYAML
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
        raise ValueError(f"Unsupported YAML format in {path} (dict or list expected).")

    wps_dict = {nm: pos for nm, pos in zip(names_list, targets_list)}
    return targets_list, names_list, wps_dict

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("expid", type=str, default="1a", help="Experiment id")
parser.add_argument("--nsteps", type=int, default=100, help="Number of steps (per phase if sequential)")
parser.add_argument("--force-training", action="store_true", help="Force trajectory regeneration")
parser.add_argument("--targets-seq", type=str, default=None,
                    help="Path to YAML with sequential targets. If set, evaluates sequential trajectory.")
args = parser.parse_args()

expid = args.expid
nsteps = args.nsteps
targets_yaml = args.targets_seq
sequential_mode = targets_yaml is not None

# Env + obstacles
envid: str = EXPERIMENTS[expid]['envid']
config: str = envid.split('_')[1]
try:
    obstacles_pos = CONFIGS[config]['obstacles']
except KeyError:
    obstacles_pos = []

print(f"Experiment: {expid}")
print(f"Obstacles position: {obstacles_pos}")

traj_path = f"{project_root()}/trajectories/trajectory_{expid}.npz"

# --- Ensure trajectory exists ---
if (not os.path.isfile(traj_path)) or args.force_training:
    if sequential_mode:
        print("Generating sequential trajectory with ReacherTrajoptSequential...")
        targets, names, wps_dict_yaml = load_targets_from_yaml(targets_yaml)
        qpos = np.array([-0.5, 0.0, 0.2])
        seq_opt = ReacherTrajoptSequential(xml_path, obstacles_pos, nsteps=nsteps, expid=expid)
        X, A, U = seq_opt.solve_sequence(targets, q0=qpos)
        phase_lengths = np.array([nsteps] * len(targets), dtype=int)
        os.makedirs(f"{project_root()}/trajectories", exist_ok=True)
        np.savez(traj_path, X=X, A=A, U=U,
                 targets=np.array(targets), phase_lengths=phase_lengths,
                 target_names=np.array(names))
    else:
        print("WARNING: --force-training used without --targets-seq; expecting existing single-target trajectory.")

# --- Load trajectory & metadata ---
if os.path.isfile(traj_path):
    print("FOUND TRAJECTORY")
    T = np.load(traj_path, allow_pickle=True)
    X = T["X"]; A = T["A"]; U = T["U"]
    has_seq = ("targets" in T.files) and ("phase_lengths" in T.files)
    if has_seq:
        targets = list(T["targets"])
        phase_lengths = T["phase_lengths"]
        names = list(T["target_names"]) if "target_names" in T.files else [f"G{i}" for i in range(1, len(targets)+1)]
        wps_dict_yaml = {nm: pos for nm, pos in zip(names, targets)}
        sequential_mode = True
    elif sequential_mode:
        # We were asked to eval sequentially but file lacks metadata → use YAML for viz/metrics
        targets, names, wps_dict_yaml = load_targets_from_yaml(targets_yaml)
        phase_lengths = np.array([nsteps] * len(targets), dtype=int)
    else:
        target_pos = np.array([*CONFIGS[config]["target"], 0.015], dtype=float)
else:
    raise FileNotFoundError(f"Trajectory not found at {traj_path}")

# --- Build robot for simulation/visualization ---
if sequential_mode:
    final_target = np.array(targets[-1], dtype=float)
    robot = RobotModel(xml_path, final_target, obstacles_pos)
else:
    robot = RobotModel(xml_path, target_pos, obstacles_pos)

mj_model = robot.mj_model
mj_data = robot.mj_data

# --- Simulate + per-step error (vs active sub-goal if sequential) ---
render_mode = "human"
frames, trajectory, accelerations, torques, errors = [], [], [], [], []
renderer = MujocoRenderer(mj_model, mj_data)
wps_dict = wps_dict_yaml if sequential_mode else {}
# Init viewer once
if render_mode == "human":
    _ = renderer.render("human")  # this creates renderer.viewer
    if sequential_mode:
        _hide_builtin_target_model(mj_model)
        _draw_waypoints_on_viewer(renderer.viewer, wps_dict)  # draw once

_hidden_builtin = False


if sequential_mode:
    phase_cum = np.cumsum(phase_lengths)
    def active_target_idx(t):
        return int(np.searchsorted(phase_cum, t, side="right"))

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
        tgt = target_pos
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

# --- Plot torques ---
plt.figure()
plt.plot(U)
plt.title("Torques over time")
plt.xlabel("Time step", fontdict={"size": 20}); plt.xticks(fontsize=15)
plt.ylabel("Torque [Nm]", fontdict={"size": 20}); plt.yticks(fontsize=15)
plt.legend([f"Joint {i+1}" for i in range(U.shape[1])], fontsize=13, loc="upper right")
plt.savefig(f"images/{expid}_trajopt.png", dpi=1000, bbox_inches="tight")

# --- Error summaries ---
errors = np.array(errors, dtype=float)
if sequential_mode:
    start = 0
    for i, L in enumerate(phase_lengths):
        seg = errors[start:start+L]
        if len(seg) > 0:
            print(f"[Phase {i}] mean_error={seg.mean():.6f}  final_error={seg[-1]:.6f}  steps={L}  target={names[i]}")
        start += L

acc_error = float(np.sum(errors) / len(errors))
print(f"ACC ERROR (overall mean): {acc_error:.6f}")
print(f"SUMMED TORQUES: {np.sum(np.sum(torques ** 2, axis=1)) * 0.01:.6f}")

renderer.close()
