import os, argparse, datetime
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import PPO, SAC
from reacher_obstacles.envs.reacher_v6 import CONFIGS

def _env_from_model_stem(stem: str) -> str:
    base = os.path.basename(stem)
    return base.split(";")[0]  # before first ';'

def _target_from_envid(envid: str) -> np.ndarray:
    key = envid.split("_")[1] if "_" in envid else None
    if not key or key not in CONFIGS:
        raise ValueError(f"Cannot derive CONFIGS key from env id '{envid}'")
    return np.array([*CONFIGS[key]["target"], 0.015])

def _rollout_and_metrics(model, envid: str, target_pos_xyz: np.ndarray, steps: int, seed: int):
    env = gym.make(envid)
    dt_attr = getattr(env.unwrapped, "dt", None)
    dt_used = float(dt_attr if dt_attr is not None else 0.01)

    U, errors = [], []

    obs, info = env.reset(seed=seed)
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # store actions
        U.append(np.asarray(action, dtype=np.float64).copy())

        # end-effector position → distance to target (for acc error)
        ee = env.unwrapped.data.body("fingertip").xpos[0:env.unwrapped.ndim]
        ee = np.array([*ee, 0.015])
        errors.append(np.linalg.norm(ee - target_pos_xyz))

        if terminated or truncated:
            break

    env.close()

    U = np.vstack(U) if len(U) else np.zeros((0, 0))
    # action-energy: Σ dt · ||u||²
    energy = float(np.sum(np.sum(U**2, axis=1)) * dt_used) if U.size else float("nan")
    acc_err = float(np.mean(errors)) if len(errors) else float("nan")
    return U, acc_err, energy

def _load_model(stem: str, algo: str, envid: str):
    algo_cls = PPO if algo.upper() == "PPO" else SAC
    pth = stem + ".pth"
    if not os.path.isfile(pth):
        raise FileNotFoundError(f"Model not found: {pth}")
    dummy_env = gym.make(envid)  # ensure spaces match the saved model
    model = algo_cls.load(pth, dummy_env)
    dummy_env.close()
    return model

# -----------------------------------------------------

def main():
    parser = argparse.ArgumentParser("Compare two models side-by-side (actions, acc error, energy).")
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "SAC"])
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--model-a", required=True, help="Model A stem (path without .pth)")
    parser.add_argument("--model-a-name", required=True, help="Model A name")
    parser.add_argument("--model-b", required=True, help="Model B stem (path without .pth)")
    parser.add_argument("--model-b-name", required=True, help="Model B name")
    args = parser.parse_args()

    os.makedirs("images", exist_ok=True)

    # derive env id (must match across models)
    envid_a = _env_from_model_stem(args.model_a)
    envid_b = _env_from_model_stem(args.model_b)
    if envid_a != envid_b:
        raise ValueError(f"Models use different env ids:\n  A: {envid_a}\n  B: {envid_b}\nCompare models trained on the same env.")
    envid = envid_a

    target_pos = _target_from_envid(envid)

    # load models and roll out
    modelA = _load_model(args.model_a, args.algo, envid)
    modelB = _load_model(args.model_b, args.algo, envid)

    U_A, acc_A, energy_A = _rollout_and_metrics(modelA, envid, target_pos, args.steps, args.seed)
    U_B, acc_B, energy_B = _rollout_and_metrics(modelB, envid, target_pos, args.steps, args.seed)

    # plot actions in [-1,1]
    nJ = U_A.shape[1]
    tA = np.arange(U_A.shape[0])
    tB = np.arange(U_B.shape[0])

    plt.figure(figsize=(14, 5))
    ax1 = plt.subplot(1, 2, 1)
    for j in range(nJ):
        ax1.plot(tA, U_A[:, j], label=f"J{j+1}")
    ax1.set_title(f"{args.model_a_name}  | Mean Error={acc_A:.3f}  Energy={energy_A:.3f}")
    ax1.set_xlabel("Time step"); ax1.set_ylabel("u ([-1,1])")
    ax1.set_ylim(-1, 1); ax1.legend(fontsize=9, loc="upper right")

    ax2 = plt.subplot(1, 2, 2)
    for j in range(nJ):
        ax2.plot(tB, U_B[:, j], label=f"J{j+1}")
    ax2.set_title(f"{args.model_b_name}  | Mean Error={acc_B:.3f}  Energy={energy_B:.3f}")
    ax2.set_xlabel("Time step"); ax2.set_ylabel("u ([-1,1])")
    ax2.set_ylim(-1, 1); ax2.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out1 = f"images/compare_u_{envid}_{args.algo}_{stamp}.png"
    plt.savefig(out1, dpi=300, bbox_inches="tight")

    print("=== RESULTS ===")
    print(f"Env: {envid} | Seed: {args.seed} | Steps: {args.steps}")
    print(f"A: {args.model_a}\n  ACC={acc_A:.6f}  Σdt||u||²={energy_A:.6f}")
    print(f"B: {args.model_b}\n  ACC={acc_B:.6f}  Σdt||u||²={energy_B:.6f}")

if __name__ == "__main__":
    main()
