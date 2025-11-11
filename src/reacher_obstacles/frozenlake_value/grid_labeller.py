from __future__ import annotations
from typing import Dict, Tuple, Set, List
from reacher_obstacles.utils import _to_r_c

class GridLabeller:
    """
    Parameters
    ----------
    waypoint_cells : Dict[str, Tuple[int,int]]
        Mapping like {"G1": (r,c), "G2": (r,c), ...} in 0..8 × 0..8.
        Names must match those used in RM conditions (e.g., near_G1).
    desc : List[str]
        The 9 lines of the FrozenLake map (each length 9, chars in {S,F,H,G}).
        Used only to emit 'hit' on 'H' cells.
    near_prefix : str
        Prefix used to build proposition names (default "near_").
    hit_label : str
        The proposition to emit when on a hole (default "hit").
    grid_n : int
        Grid size (default 9).

    def label(): Returns the SAME proposition the RM expects, e.g. "near_G1", "hit".
    """

    def __init__(
            self,
            waypoint_cells: Dict[str, Tuple[int, int]],
            desc: List[str],
            near_prefix: str = "near_",
            hit_label: str = "hit",
            grid_n: int = 9,
            continuous: bool = False
    ):
        self.waypoint_cells = dict(waypoint_cells)
        self.desc = list(desc)
        self.near_prefix = str(near_prefix)
        self.hit_label = str(hit_label)
        self.n = int(grid_n)
        self.continuous = bool(continuous)

    def _current_rc(self, env) -> Tuple[int, int]:
        """
        Robustly fetch (r,c) from FrozenLake, even if wrapped.
        """

        # Try the common paths to reach the raw FL env
        base = getattr(env, "unwrapped", env)
        if not hasattr(base, "s"):
            # RMWrapper or others may add .env layers
            cur = getattr(env, "env", None)
            while cur is not None and not hasattr(getattr(cur, "unwrapped", cur), "s"):
                cur = getattr(cur, "env", None)
            base = getattr(cur, "unwrapped", cur) if cur is not None else base

        s = getattr(base, "s", None)
        if s is None:
            # As last resort, let caller fail loudly
            raise AttributeError("GridLabeller: cannot access FrozenLake 's' state.")
        r, c = divmod(int(s), self.n)
        return r, c

    def label(self, env) -> Set[str]:
        r, c = self._current_rc(env)
        labels: Set[str] = set()

    # Collision / hole label
        try:
            if self.desc[r][c] == "H":
                labels.add(self.hit_label)
        except Exception:
            pass  # if desc not aligned, silently skip 'hit'

        # near_* labels
        for name, (ri, ci) in self.waypoint_cells.items():
            if self.continuous:
                ri, ci = _to_r_c([ri, ci])
            if r == ri and c == ci:
                labels.add(f"{self.near_prefix}{name}")

        return labels
