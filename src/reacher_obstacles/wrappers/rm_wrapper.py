from __future__ import annotations
from typing import Tuple, Dict, Any, List
import numpy as np
import gymnasium as gym

from reacher_obstacles.rm.reward_machine import RewardMachine
from reacher_obstacles.rm.labeller import Labeller

class RMWrapper(gym.Wrapper):
    """
    Cross-product baseline:
      - reward_mode="replace": reward = w_rm * r_rm (strict baseline)
      - reward_mode="add":     reward = r_env + w_rm * r_rm (non-baseline variant)
    Optionally, if the env exposes a way to set the active target, route it.
    """
    def __init__(self, env, rm, labeller, w_rm=1.0, route_target=False, waypoint_order=None,
                 reward_mode: str = "replace"):  # "replace" (baseline) or "add"
        super().__init__(env)
        self.rm = rm
        self.labeller = labeller
        self.w_rm = float(w_rm)
        self.route_target = bool(route_target)
        self.u_to_wp = waypoint_order or {}  # map RM state -> waypoint name (e.g., {"u1":"G1", ...})
        self.reward_mode = reward_mode

        # augment observation space
        orig = self.observation_space
        k = len(self.rm.states)
        self._onehot_len = k
        low = np.concatenate([orig.low, np.zeros(k, dtype=orig.low.dtype)])
        high = np.concatenate([orig.high, np.ones(k, dtype=orig.high.dtype)])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=orig.dtype)

    def reset(self, **kwargs):
        # 1) RM to u0
        self.rm.reset()

        # 2) route the env target to the u0 waypoint before the inner reset,
        #    so RH sees the correct goal during its own reset
        if self.route_target:
            self._route_target()

        # 3) reset the inner env (RH will now rebuild its map around the routed goal).
        obs, info = self.env.reset(**kwargs)

        # 4) re-do routing in case env reset the original target
        if self.route_target:
            self._route_target()

        return self._augment_obs(obs), info


    def step(self, action):
        obs, r_env, terminated, truncated, info = self.env.step(action)

        sigma = set(self.labeller.label(self.env))
        # expose collisions to RM
        if info.get("hit", 0):
            sigma.add("hit")
        sigma = list(sigma)
        # remember RM state
        prev_u = self.rm.u
        _, r_rm = self.rm.step(sigma)

        # (visual only) route when RM state changes
        routed_to = None
        if self.route_target and (self.rm.u != prev_u):   # <-- NEW
                self._route_target()                     # <-- NEW
                routed_to = self.u_to_wp.get(self.rm.u)


    # strict CP baseline: ignore environment reward
        if self.reward_mode == "replace":
            r_total = float(self.w_rm * r_rm)
        else:  # "add" = non-baseline variant that includes env reward too
            r_total = float(r_env + self.w_rm * r_rm)

        # compute distance to CURRENT waypoint (for logging/diagnostics)
        wp_name = self.u_to_wp.get(self.rm.u) if hasattr(self, "u_to_wp") else None
        wp_pos = self.labeller.waypoints.get(wp_name) if wp_name else None
        try:
            ee = self.env.unwrapped.data.body("fingertip").xpos[:self.env.unwrapped.ndim]
            ee = np.array([*ee, 0.015], dtype=float)
            dist_wp = float(np.linalg.norm(ee - np.asarray(wp_pos, float))) if wp_pos is not None else float("nan")
        except Exception:
            dist_wp = float("nan")

        # store info for TB/debug
        info = dict(info or {})
        info["env_reward"]   = float(r_env)
        info["rm_reward"]    = float(r_rm)
        info["total_reward"] = float(r_total)
        info["rm_state"]     = self.rm.u
        info["sigma"]        = list(sigma)
        info["wp_name"]      = wp_name
        info["wp_pos"]       = (wp_pos.tolist() if isinstance(wp_pos, np.ndarray) else wp_pos)
        info["dist_to_wp"]   = dist_wp
        info["r_mode"]       = self.reward_mode
        if routed_to is not None:
            info["routed_to"] = routed_to  # only present on steps where we routed

        done = terminated or truncated or self.rm.is_terminal()
        return self._augment_obs(obs), r_total, done, truncated and not done, info

    # helpers
    def _augment_obs(self, obs):
        onehot = np.zeros(self._onehot_len, dtype=self.observation_space.dtype)
        idx = self._state_index(self.rm.u)
        onehot[idx] = 1.0
        return np.concatenate([obs, onehot], axis=-1)

    def _state_index(self, u: str) -> int:
        try:
            return self.rm.states.index(u)
        except ValueError:
            return 0

    def _route_target(self):
        """Optional: set env's current target to waypoint associated with current RM state."""
        if not self.route_target:
            return
        wp_name = self.u_to_wp.get(self.rm.u)
        if not wp_name:
            return
        # try a few common hooks without hardcoding env
        pos = self.labeller.waypoints.get(wp_name)
        if pos is None:
            return
        un = self.env.unwrapped
        un.set_target(pos)