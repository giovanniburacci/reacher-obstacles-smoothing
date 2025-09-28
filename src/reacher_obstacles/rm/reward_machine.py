# reward_machine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import json, os

try:
    import yaml  # optional
except Exception:
    yaml = None


@dataclass(frozen=True)
class Edge:
    src: str
    cond: str            # boolean over atomic props or single atom (e.g., "near_G1 && !collision")
    dst: str
    reward: float = 0.0


class RewardMachine:
    def __init__(self, states: List[str], initial: str, accepting: Set[str], edges: List[Edge]):
        self.states = states
        self.initial = initial
        self.accepting = set(accepting)
        self.edges_by_src: Dict[str, List[Edge]] = {}
        for e in edges:
            self.edges_by_src.setdefault(e.src, []).append(e)
        self._u = initial

    @property
    def u(self) -> str:
        return self._u

    def reset(self) -> str:
        self._u = self.initial
        return self._u

    def is_terminal(self) -> bool:
        return self._u in self.accepting

    def step(self, sigma: Set[str]) -> Tuple[str, float]:
        """advance given a set of true atomic propositions sigma; returns (new_u, rm_reward)."""
        edges = self.edges_by_src.get(self._u, [])
        for e in edges:
            if _eval_cond(e.cond, sigma):
                self._u = e.dst
                return self._u, float(e.reward)
        return self._u, 0.0


def _eval_cond(expr: str, sigma: Set[str]) -> bool:
    """very small boolean evaluator over atoms in sigma: &&, ||, !, parentheses"""
    if not expr or expr.lower() in ("true", "1"):
        return True
    # tokenize simple operators
    repl = (("&&", " and "), ("||", " or "), ("!", " not "))
    s = expr
    for a, b in repl:
        s = s.replace(a, b)
    # allowed names are atoms in sigma; others evaluate to False
    names = {k: (k in sigma) for k in _atoms(expr)}
    return bool(eval(s, {"__builtins__": {}}, names))


def _atoms(expr: str) -> Set[str]:
    out, tok, in_name = set(), [], False
    for ch in expr:
        if ch.isalnum() or ch in "._":
            tok.append(ch); in_name = True
        else:
            if in_name:
                out.add("".join(tok)); tok.clear()
            in_name = False
    if tok: out.add("".join(tok))
    # filter out common operator words if they slipped in
    return {a for a in out if a not in {"and", "or", "not", "true", "false"}}


def load_rm_spec(path: str) -> RewardMachine:
    with open(path, "r") as f:
        if path.lower().endswith((".yml", ".yaml")) and yaml is not None:
            spec = yaml.safe_load(f)
        else:
            spec = json.load(f)
    states = spec["states"]
    initial = spec["initial"]
    accepting = set(spec.get("accepting", []))
    edges = [Edge(**e) for e in spec["transitions"]]
    return RewardMachine(states, initial, accepting, edges)
