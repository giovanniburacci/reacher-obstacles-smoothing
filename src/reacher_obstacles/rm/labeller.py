# labeller.py
from __future__ import annotations
from typing import Dict, List, Set, Tuple
import numpy as np

class Labeller:
    """
    Builds atomic propositions based on EE proximity to waypoints and collisions.
    waypoints: dict name -> xyz (np.array shape (3,))
    epsilons: dict name -> float radius (defaults to global_eps if missing)
    """
    def __init__(self, waypoints: Dict[str, np.ndarray], global_eps: float = 0.03, epsilons: Dict[str, float] = None):
        self.waypoints = {k: np.asarray(v, float).reshape(3) for k, v in waypoints.items()}
        self.global_eps = float(global_eps)
        self.epsilons = {**({} if epsilons is None else epsilons)}

    def label(self, env) -> Set[str]:
        """Extract propositions from current unwrapped env state."""
        sigma: Set[str] = set()
        ee = env.unwrapped.data.body("fingertip").xpos[:env.unwrapped.ndim]
        ee = np.array([*ee, 0.015], dtype=float)

        for name, pos in self.waypoints.items():
            eps = float(self.epsilons.get(name, self.global_eps))
            if np.linalg.norm(ee - pos) <= eps:
                sigma.add(f"near_{name}")

        # Optional: obstacle contact flag if available
        try:
            contact = bool(getattr(env.unwrapped, "episode_hit", False))
            if contact:
                sigma.add("collision")
        except Exception:
            pass

        return sigma
