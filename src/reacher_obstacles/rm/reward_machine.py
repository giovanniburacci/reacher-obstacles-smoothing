# reward_machine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import json, os

try:
    import yaml  # optional dependency for YAML RM specs
except Exception:
    yaml = None


@dataclass(frozen=True)
class Edge:
    src: str
    cond: str        # boolean formula over atomic props (e.g. "near_G1 && !hit")
    dst: str
    reward: float = 0.0


class RewardMachine:
    def __init__(self, states: List[str], initial: str,
                 accepting: Set[str], edges: List[Edge]):
        self.states = states
        self.initial = initial
        self.accepting = set(accepting)

        # build quick lookup table: src_state -> list of outgoing edges
        self.edges_by_src: Dict[str, List[Edge]] = {}
        for e in edges:
            self.edges_by_src.setdefault(e.src, []).append(e)

        # current RM state (u_t)
        self._u = initial

    @property
    def u(self) -> str:
        return self._u

    def reset(self) -> str:
        # reset automaton to initial state (called at episode start)
        self._u = self.initial
        return self._u

    def is_terminal(self) -> bool:
        # check if current RM state is in the accepting set
        return self._u in self.accepting

    def step(self, sigma: Set[str]) -> Tuple[str, float]:
        """
        Advance RM one step given the set of active propositions σ.
        For each outgoing edge from current state, evaluate its condition.
        First matching edge is taken; RM moves to its destination and returns
        the edge reward. If none match, stay in current state and return 0.
        """
        edges = self.edges_by_src.get(self._u, [])
        for e in edges:
            if _eval_cond(e.cond, sigma):      # condition satisfied
                self._u = e.dst                # transition to new state
                return self._u, float(e.reward)
        # no transition fired
        return self._u, 0.0


def _eval_cond(expr: str, sigma: Set[str]) -> bool:
    """
    Evaluate a small boolean expression over the active propositions in σ.
    Supports &&, ||, ! and parentheses.
    Example:
        expr = "near_G1 && !collision"
        sigma = {"near_G1"}  -> True
        sigma = {"collision"} -> False
    """
    if not expr or expr.lower() in ("true", "1"):
        return True

    # translate C-style boolean ops to Python syntax
    s = expr
    for a, b in (("&&", " and "), ("||", " or "), ("!", " not ")):
        s = s.replace(a, b)

    # build a dict like {"near_G1": True, "collision": False, ...}
    names = {k: (k in sigma) for k in _atoms(expr)}

    # evaluate expression safely:
    #   - no builtins, so eval can't call anything dangerous
    #   - 'names' provides only boolean variables relevant to expr
    return bool(eval(s, {"__builtins__": {}}, names))


def _atoms(expr: str) -> Set[str]:
    """
    Extract all atomic proposition names appearing in expr.
    Simple tokenizer that walks characters and collects sequences
    of letters/numbers/underscores. Returns a set of strings.
    Example:
        "near_G1 && !collision" -> {"near_G1", "collision"}
    """
    out, tok, in_name = set(), [], False
    for ch in expr:
        if ch.isalnum() or ch in "._":
            # inside a token (variable name)
            tok.append(ch)
            in_name = True
        else:
            # reached a separator, close current token if any
            if in_name:
                out.add("".join(tok))
                tok.clear()
            in_name = False
    if tok:
        out.add("".join(tok))

    # drop reserved words that might appear by coincidence
    return {a for a in out if a not in {"and", "or", "not", "true", "false"}}


def load_rm_spec(path: str) -> RewardMachine:
    """
    Load a RewardMachine definition from a JSON or YAML file.
    Expected keys:
        states, initial, accepting (optional), transitions
    where transitions are dicts {src, cond, dst, reward}.
    """
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
