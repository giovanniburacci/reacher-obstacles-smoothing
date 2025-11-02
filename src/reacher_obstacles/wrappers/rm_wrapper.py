from __future__ import annotations
from typing import Tuple, Dict, Any, List
import numpy as np
import gymnasium as gym

from reacher_obstacles.rm.reward_machine import RewardMachine
from reacher_obstacles.rm.labeller import Labeller


class RMWrapper(gym.Wrapper):
    """
    Cross-product baseline between environment and Reward Machine.

      - reward_mode="replace": reward = w_rm * r_rm  (strict baseline)
      - reward_mode="add":     reward = r_env + w_rm * r_rm  (non-baseline variant)

    Optionally, if the env exposes a method to set the active target, route it
    automatically whenever the RM changes state.
    """

    def __init__(self, env, rm, labeller, w_rm=1.0, route_target=False,
                 waypoint_order=None, reward_mode: str = "replace", waypoints: Dict[str, Tuple[float, float, float]] = {}):
        super().__init__(env)
        self.rm = rm
        self.labeller = labeller
        self.w_rm = float(w_rm)
        self.route_target = bool(route_target)
        self.u_to_wp = waypoint_order or {}  # map RM state -> waypoint name (e.g. {"u1":"G1"})
        self.waypoints = waypoints
        self.env.unwrapped.waypoints = self.waypoints  # expose to env for logging if needed
        self.reward_mode = reward_mode

        # extend observation space with one-hot encoding of RM state
        orig = self.observation_space
        k = len(self.rm.states)
        self._onehot_len = k
        low = np.concatenate([orig.low, np.zeros(k, dtype=orig.low.dtype)])
        high = np.concatenate([orig.high, np.ones(k, dtype=orig.high.dtype)])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=orig.dtype)

    def reset(self, **kwargs):
        # 1) reset RM to its initial state
        self.rm.reset()

        # 2) if routing is enabled, set env target to the waypoint of u0
        #    (important: do this before env.reset so inner env sees correct goal)
        if self.route_target:
            self._route_target()

        # 3) reset underlying env (may rebuild internal maps or heuristics)
        obs, info = self.env.reset(**kwargs)

        # 4) re-apply routing in case env.reset() overwrote the target
        if self.route_target:
            self._route_target()

        return self._augment_obs(obs), info

    def step(self, action):
        # step environment normally
        obs, r_env, terminated, truncated, info = self.env.step(action)

        # build proposition set σ from labeller
        sigma = set(self.labeller.label(self.env))

        # propagate "hit" flag from env info if present
        if info.get("hit", 0):
            sigma.add("hit")

        prev_u = self.rm.u
        _, r_rm = self.rm.step(sigma)  # advance RM using σ

        # if RM state changed and routing is active, update target
        routed_to = None
        if self.route_target and (self.rm.u != prev_u):
            self._route_target()
            routed_to = self.u_to_wp.get(self.rm.u)

        # compute final reward depending on mode
        if self.reward_mode == "replace":
            # strict cross-product baseline: ignore env reward
            r_total = float(self.w_rm * r_rm)
            r_total = float(r_env)
        else:
            # additive variant: keep env reward too
            r_total = float(r_env + self.w_rm * r_rm)

        # diagnostic: compute distance to current waypoint (for logs)
        wp_name = self.u_to_wp.get(self.rm.u) if hasattr(self, "u_to_wp") else None
        wp_pos = self.labeller.waypoints.get(wp_name) if wp_name else None
        try:
            ee = self.env.unwrapped.data.body("fingertip").xpos[:self.env.unwrapped.ndim]
            ee = np.array([*ee, 0.015], dtype=float)
            dist_wp = float(np.linalg.norm(ee - np.asarray(wp_pos, float))) if wp_pos is not None else float("nan")
        except Exception:
            dist_wp = float("nan")

        # collect full info dict for logging/debugging
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
            info["routed_to"] = routed_to  # only present when routing occurred

        # terminate if env done, truncated, or RM reached accepting state
        # done = terminated or truncated or self.rm.is_terminal()
        done = truncated
        # truncated and not done is kept for proper Gymnasium return signature
        return self._augment_obs(obs), r_total, truncated, truncated and not done, info

    # -------------------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------------------

    def _augment_obs(self, obs):
        """
        Concatenate a one-hot encoding of the current RM state to the base obs.
        Allows the policy to condition actions on task phase (u_t).
        """
        onehot = np.zeros(self._onehot_len, dtype=self.observation_space.dtype)
        idx = self._state_index(self.rm.u)
        onehot[idx] = 1.0
        return np.concatenate([obs, onehot], axis=-1)

    def _state_index(self, u: str) -> int:
        # map state name -> index in self.rm.states
        try:
            return self.rm.states.index(u)
        except ValueError:
            return 0

    def _route_target(self):
        """
        If routing is enabled and mapping exists, set the env's current
        target to the waypoint corresponding to the active RM state.
        This keeps multi-goal tasks synchronized between RM and env.
        """
        if not self.route_target:
            return
        wp_name = self.u_to_wp.get(self.rm.u)
        if not wp_name:
            return
        pos = self.labeller.waypoints.get(wp_name)
        if pos is None:
            return
        un = self.env.unwrapped
        un.set_target(pos)
