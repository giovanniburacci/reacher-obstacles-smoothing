# labeller.py
from __future__ import annotations
from typing import Dict, List, Set, Tuple
import numpy as np


class Labeller:
    """
    Computes the set of active atomic propositions σ from the environment.
    A proposition is typically "near_<WAYPOINT>" if the end effector is within
    a given radius of that waypoint, and optionally "collision" if a contact
    occurred.  Used by the RewardMachine to decide which transition to fire.
    """

    def __init__(self,
                 waypoints: Dict[str, np.ndarray],
                 global_eps: float = 0.03,
                 epsilons: Dict[str, float] = None):
        # normalize waypoint positions to (3,) float arrays
        self.waypoints = {k: np.asarray(v, float).reshape(3)
                          for k, v in waypoints.items()}

        self.global_eps = float(global_eps)

        # per-waypoint radii override global_eps if provided
        self.epsilons = {**({} if epsilons is None else epsilons)}

    def label(self, env) -> Set[str]:
        """
        Extract the current set of true propositions σ from the environment.
        Called once per environment step.
        """
        sigma: Set[str] = set()

        # fingertip Cartesian position (x,y,z)
        ee = env.unwrapped.data.body("fingertip").xpos[:env.unwrapped.ndim]

        # small z-offset (specific to this Mujoco model; keeps consistent scale)
        ee = np.array([*ee, 0.015], dtype=float)

        # check proximity to each waypoint
        for name, pos in self.waypoints.items():
            eps = float(self.epsilons.get(name, self.global_eps))
            # if EE within epsilon distance -> activate "near_<waypoint>"
            if np.linalg.norm(ee - pos) <= eps:
                sigma.add(f"near_{name}")

        # add "collision" if env reports a hit flag (optional)
        try:
            contact = bool(getattr(env.unwrapped, "episode_hit", False))
            if contact:
                sigma.add("collision")
        except Exception:
            # some envs may not expose 'episode_hit'; ignore silently
            pass

        return sigma
