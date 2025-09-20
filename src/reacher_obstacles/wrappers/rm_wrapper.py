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
        self.rm.reset()
        obs, info = self.env.reset(**kwargs)
        self._maybe_route_target()
        return self._augment_obs(obs), info

    def step(self, action):
        obs, r_env, terminated, truncated, info = self.env.step(action)

        sigma = set(self.labeller.label(self.env))
        if info.get("hit", 0):                       # <-- NEW: expose collisions to RM
            sigma.add("hit")
        sigma = list(sigma)
        prev_u = self.rm.u                     # <-- NEW: remember RM state
        _, r_rm = self.rm.step(sigma)

        # (visual only) route when RM state changes
        if self.route_target and (self.rm.u != prev_u):   # <-- NEW
            self._maybe_route_target()                     # <-- NEW

        # strict CP baseline: ignore environment reward
        if self.reward_mode == "replace":
            r_total = float(self.w_rm * r_rm)
        else:  # "add" = non-baseline variant that includes env reward too
            r_total = float(r_env + self.w_rm * r_rm)

        # bookkeeping for logs
        info = dict(info or {})
        info["env_reward"] = float(r_env)
        info["rm_reward"] = float(r_rm)
        info["rm_state"] = self.rm.u
        info["sigma"] = list(sigma)

        done = terminated or truncated or self.rm.is_terminal()
        return self._augment_obs(obs), r_total, done, truncated and not done, info

    # ----- helpers
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

    def _maybe_route_target(self):
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
        for attr in ("set_target", "set_goal", "set_waypoint"):
            if hasattr(un, attr):
                try:
                    getattr(un, attr)(pos)  # e.g., env.unwrapped.set_target(np.array([x,y,z]))
                    print(f"[RM route] u={self.rm.u} -> {wp_name}")
                    return
                except Exception:
                    pass
        # fallback: common attribute names
        for name in ("target", "goal", "waypoint"):
            if hasattr(un, name):
                try:
                    setattr(un, name, pos)
                    return
                except Exception:
                    pass
