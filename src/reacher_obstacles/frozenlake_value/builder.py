from __future__ import annotations
import copy
import time
import numpy as np
import gymnasium as gym
from typing import List, Tuple

from reacher_obstacles.wrappers.rm_wrapper import RMWrapper
from reacher_obstacles.rm.reward_machine import RewardMachine
from reacher_obstacles.frozenlake_value.grid_labeller import GridLabeller

GridDesc = List[str]  # list of 9 strings of length 9 with chars in {S,F,H,G}

def desc_from_gmap(gmap: np.ndarray, waypoints) -> GridDesc:
    """
    Convert reacher_rh gmap (-1 free, >-1 obstacle) to a FrozenLake desc.
    S = top-left free cell, G = bottom-left free cell (placeholders; RM gives reward).
    """
    assert gmap.shape == (9, 9)
    desc = np.full((9, 9), 'F', dtype='<U1')
    desc[gmap == 18] = 'H'
    desc[4,4] = 'H'
    desc[4,7] = 'S'
    desc[8,0] = 'G'

    H, W = desc.shape

    holes = [(r, c) for r in range(9) for c in range(9) if desc[r, c] == 'H' and not(r == 4 and c == 4)]
    for r, c in holes:
        dists = {
            "down":  H - 1 - r,
            "up":    r,
            "right": W - 1 - c,
            "left":  c,
        }
        # tie-break preference: down > up > right > left
        # order = ["down", "up", "right", "left"]
        # direction = min(order, key=lambda k: (dists[k], order.index(k)))

        # if direction == "down":
        #     for rr in range(r, H):        desc[rr, c] = 'H'
        # elif direction == "up":
        #     for rr in range(0, r + 1):    desc[rr, c] = 'H'
        # elif direction == "right":
        #     for cc in range(c, W):        desc[r, cc] = 'H'
        # elif direction == "left":
        #     for cc in range(0, c + 1):    desc[r, cc] = 'H'

    # TODO uncomment to have multiple goals but non working
    # for wp_name, (r, c) in waypoints.items():
    #     if wp_name.startswith('G'):
    #         desc[r, c] = 'G'


    # fallbacks if grid is degenerate (no free cells)
    if 'S' not in desc or 'G' not in desc:
        desc = np.full((9, 9), 'F', dtype='<U1')
        desc[0, 0] = 'S'
        desc[8, 8] = 'G'

    return [''.join(desc[r, :].tolist()) for r in range(9)]


def _rm_clone(rm: RewardMachine) -> RewardMachine:
    return copy.deepcopy(rm)


def _obs_to_rc_from_env(env, n=9) -> tuple[int, int]:
    """Read FrozenLake raw integer state robustly even if obs was augmented."""
    base = getattr(env, "env", None)
    while base is not None and not hasattr(getattr(base, "unwrapped", base), "s"):
        base = getattr(base, "env", None)
    raw = getattr(getattr(base, "unwrapped", base), "s", None)
    if raw is None:
        raise RuntimeError("Cannot access FrozenLake internal state 's'")
    r, c = divmod(int(raw), n)
    return r, c

def _greedy_replay(env, Q: np.ndarray, episodes: int, max_steps: int,
                   render: bool = False):
    rm_idx = {u: i for i, u in enumerate(env.rm.states)}
    lengths = []
    for ep in range(episodes):
        print(f"  episode {ep}/{episodes}")
        obs, info = env.reset()
        terminated = truncated = False
        steps = 0
        if render:
            env.render()
        while not (terminated or truncated):
            r, c = _obs_to_rc_from_env(env)
            k = rm_idx[env.rm.u]
            a = int(np.argmax(Q[r, c, k]))
            obs, rew, terminated, truncated, info = env.step(a)
            if render:
                env.render()
                time.sleep(0.25)
            steps += 1
            if steps >= max_steps:
                truncated = True
        lengths.append(steps)

    env.close()



def learn_v_table(
        gmap: np.ndarray,
        rm,
        waypoint_cells_9x9: dict[str, Tuple[int, int]],
        gamma: float = 0.98,
        episodes: int = 30000,
        max_steps: int = 100,
        eps_start: float = 0.3,
        eps_final: float = 0.01,
        alpha_start: float = 0.5,
        alpha_final: float = 0.05,
        do_replay: bool = True,
        replay_episodes: int = 10,
        render_replay: bool = False,
        replay_sleep_s: float = 0.25,
) -> Tuple[np.ndarray, List[str]]:
    """
    Train and optionally replay on a separate rendered env.
    Returns:
        V: np.ndarray [K,9,9]
        rm_state_order: list[str]
    """
    desc = desc_from_gmap(gmap, waypoint_cells_9x9)
    labeller = GridLabeller(
        waypoint_cells=waypoint_cells_9x9,
        desc=desc,
        near_prefix="near_",
        hit_label="hit",
        grid_n=9,
    )
    print(f"desc: {desc}")

    fl_train = gym.make("FrozenLake-v1", desc=desc, is_slippery=False)
    env_train = RMWrapper(
        fl_train,
        rm=copy.deepcopy(rm),
        labeller=labeller,
        w_rm=1.0,
        route_target=False,
        reward_mode="replace",
    )

    K = len(rm.states)
    Q = np.zeros((9, 9, K, 4), dtype=np.float32)  # 0:L,1:D,2:R,3:U
    rm_state_index = {u: i for i, u in enumerate(rm.states)}

    def epsilon(t): return eps_final + (eps_start - eps_final) * np.exp(-t / (episodes / 4))
    def alpha(t):   return alpha_final + (alpha_start - alpha_final) * np.exp(-t / (episodes / 3))

    t = 0
    for ep in range(episodes):
        _, _ = env_train.reset()
        if ep % 5000 == 0 and ep > 0:
            print(f"[FL V-builder] Episode {ep}/{episodes} (t={t})", flush=True)
        for _ in range(max_steps):
            r0, c0 = _obs_to_rc_from_env(env_train)
            k0 = rm_state_index[env_train.rm.u]
            a = env_train.action_space.sample() if np.random.rand() < epsilon(t) else int(np.argmax(Q[r0, c0, k0]))
            obs2, r, term, trunc, info2 = env_train.step(a)

            r1, c1 = _obs_to_rc_from_env(env_train)
            k1 = rm_state_index[env_train.rm.u]
            target = r + (0.0 if (term or trunc) else gamma * np.max(Q[r1, c1, k1]))
            Q[r0, c0, k0, a] += alpha(t) * (target - Q[r0, c0, k0, a])
            t += 1

            if term or trunc:
                break

    # Build V
    V = np.max(Q, axis=3)           # [9,9,K]
    V = np.transpose(V, (2, 0, 1))  # [K,9,9]

    ### set v = 1.0

    k_pre = rm_state_index['u2']
    r_goal, c_goal = labeller.waypoint_cells['G3']  # tuple (r, c)

    # Apply the override
    print(f"[FL] override V[{k_pre}, {r_goal}, {c_goal}] = 1.0", flush=True)

    V[k_pre, r_goal, c_goal] = 1.0

    # ---------- OPTIONAL RENDERED REPLAY ----------
    if do_replay:
        fl_replay = gym.make(
            "FrozenLake-v1",
            desc=desc,
            is_slippery=False,
            render_mode="human"
        )
        env_replay = RMWrapper(
            fl_replay,
            rm=copy.deepcopy(rm),
            labeller=labeller,
            w_rm=1.0,
            route_target=False,
            reward_mode="replace",
        )
        print(f"Q: {Q}")
        print(f"V: {V}")
        _greedy_replay(env_replay, Q, episodes=replay_episodes, max_steps=max_steps, render=render_replay)
        try:
            env_replay.close()
        except Exception:
            pass

    try:
        env_train.close()
    except Exception:
        pass

    return V, list(rm.states), Q
