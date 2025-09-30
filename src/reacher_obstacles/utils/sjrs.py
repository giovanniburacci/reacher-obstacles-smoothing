from typing import Optional, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import Wrapper
from stable_baselines3.common.callbacks import BaseCallback


# =========================
#  SJRS reward wrapper
#
# =========================
class SJRSRewardWrapper(Wrapper):
    def __init__(
            self,
            env: gym.Env,
            lambda_t: float = 0.0,
            lambda_a: float = 0.0,
            lambda_c: float = 0.0,
            sjrs_mode: str = "torque",  # "torque" | "action"
    ):
        super().__init__(env)
        self.lambda_t = float(lambda_t)
        self.lambda_a = float(lambda_a)
        self.lambda_c = float(lambda_c)
        self.sjrs_mode = str(sjrs_mode)

        # cache actuator count
        self.n_act = int(getattr(env.unwrapped.model, "nu", 0)) or None

        # histories (persist across steps; reset in reset())
        self._last_vec: Optional[np.ndarray] = None
        self._last_dvec: Optional[np.ndarray] = None
        self._last_tau: Optional[np.ndarray] = None
        self._last_dtau: Optional[np.ndarray] = None

    # ---------- helpers ----------
    def _read_tau(self) -> np.ndarray:
        tau = np.asarray(self.env.unwrapped.data.qfrc_actuator, dtype=np.float64)
        return tau[: self.n_act] if self.n_act is not None else tau

    # ---------- gym API ----------
    def reset(self, **kwargs):
        # clear history
        self._last_vec = None
        self._last_dvec = None
        self._last_tau = None
        self._last_dtau = None
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # --- reference torques (for sanity only; not used for penalty when mode="action") ---
        tau = self._read_tau()
        if self._last_tau is None:
            dtau = np.zeros_like(tau)
            d2tau = np.zeros_like(tau)
        else:
            dtau = tau - self._last_tau
            d2tau = dtau - (self._last_dtau if self._last_dtau is not None else 0.0)

        # --- choose the penalized signal vec ---
        if self.sjrs_mode == "torque":
            vec = tau.astype(np.float64, copy=True)
        else:  # "action"
            vec = np.asarray(action, dtype=np.float64).copy()

        # --- first & second differences (use prior history; update AFTER logging) ---
        if self._last_vec is None:
            dvec = np.zeros_like(vec)
            d2vec = np.zeros_like(vec)
        else:
            dvec = vec - self._last_vec
            d2vec = dvec - (self._last_dvec if self._last_dvec is not None else 0.0)

        # --- RAW components (λ-independent) ---
        E_step = float(np.dot(vec, vec))        # magnitude
        V_step = float(np.dot(dvec, dvec))      # temporal (Δ)
        C_step = float(np.dot(d2vec, d2vec))    # curvature (Δ²)

        # --- penalty (unchanged) & reward edit ---
        penalty_step = (self.lambda_a * E_step
                        + self.lambda_t * V_step
                        + self.lambda_c * C_step)
        reward = float(reward) - float(penalty_step)

        # --- clean nested logging for TensorBoard ---
        s: Dict[str, Any] = {
            "mode": self.sjrs_mode,
            # per-step raw components (use these to tune λ)
            "E_step": E_step,
            "V_step": V_step,
            "C_step": C_step,
            "penalty_step": float(penalty_step),
            # tiny norms of the chosen signal for sanity
            "vec_norm": float(np.linalg.norm(vec)),
            "dvec_norm": float(np.linalg.norm(dvec)),
            "d2vec_norm": float(np.linalg.norm(d2vec)),
            # lambdas (so TB records what was used)
            "lambda_a": float(self.lambda_a),
            "lambda_t": float(self.lambda_t),
            "lambda_c": float(self.lambda_c),
            "active": 1.0,
        }

        # Backwards-compatible aliases (keep existing callback working)
        s["magnitude_raw"] = s["E_step"]
        s["temporal_raw"] = s["V_step"]
        s["curvature_raw"] = s["C_step"]
        s["penalty"] = s["penalty_step"]

        info = dict(info or {})
        info["sjrs"] = s  # single source of truth

        # --- update histories (after computing deltas) ---
        self._last_vec = vec
        self._last_dvec = dvec
        self._last_tau = tau
        self._last_dtau = dtau

        return obs, reward, terminated, truncated, info


# =========================
#  TensorBoard callback
#  (episode sums + interval stats for E/V/C)
# =========================
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class SJRSCallback(BaseCallback):
    """
    Logs:
      Per-episode: sjrs/E_sum, V_sum, C_sum, penalty_sum, steps_ep
      Interval   : sjrs/E_step_mean, V_step_mean, C_step_mean,
                   sjrs/E_step_p95,  V_step_p95,  C_step_p95,
                   sjrs/penalty_step_mean
      Lambdas    : sjrs/lambda_a_mean, lambda_t_mean, lambda_c_mean
      Heartbeat  : sjrs/timesteps, sjrs/alive (1)
    """
    def __init__(self, prefix: str = "sjrs", interval: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.prefix = prefix
        self.interval = int(interval)
        self._sum = {}  # per-env episode accumulators
        self._buf = {
            "E_step": [], "V_step": [], "C_step": [], "penalty_step": [],
            "lambda_a": [], "lambda_t": [], "lambda_c": [], "active": [],
        }

    def _init_sum(self, i: int):
        self._sum[i] = dict(E=0.0, V=0.0, C=0.0, P=0.0, steps=0)

    def _record_interval(self):
        # means
        for k in ("E_step", "V_step", "C_step", "penalty_step",
                  "lambda_a", "lambda_t", "lambda_c", "active"):
            if self._buf[k]:
                self.logger.record(f"{self.prefix}/{k}_mean", float(np.mean(self._buf[k])))
        # p95 for raw components
        for k in ("E_step", "V_step", "C_step"):
            if self._buf[k]:
                self.logger.record(f"{self.prefix}/{k}_p95", float(np.percentile(self._buf[k], 95)))
        # heartbeat
        self.logger.record(f"{self.prefix}/timesteps", self.num_timesteps)
        self.logger.dump(self.model.num_timesteps)
        # clear
        for k in self._buf:
            self._buf[k].clear()

    # --- SB3 hooks ---
    def _on_training_start(self) -> None:
        # Use a numeric flag instead of record_text (not available in your SB3)
        self.logger.record(f"{self.prefix}/alive", 1.0)
        self.logger.dump(self.model.num_timesteps)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", None)

        for i, info in enumerate(infos):
            sj = info.get(self.prefix)
            if not isinstance(sj, dict):
                continue

            # init per-env episode sums
            if i not in self._sum:
                self._init_sum(i)

            # accumulate episode sums
            self._sum[i]["E"] += float(sj.get("E_step", 0.0))
            self._sum[i]["V"] += float(sj.get("V_step", 0.0))
            self._sum[i]["C"] += float(sj.get("C_step", 0.0))
            self._sum[i]["P"] += float(sj.get("penalty_step", 0.0))
            self._sum[i]["steps"] += 1

            # interval buffers
            for k in self._buf:
                self._buf[k].append(float(sj.get(k, 0.0)))

            # robust episode-end detection (VecEnv / Gymnasium)
            if dones is not None:
                done_i = bool(dones[i]) if hasattr(dones, "__len__") else bool(dones)
            else:
                done_i = False
            ep_over = done_i or ("episode" in info) or ("terminal_observation" in info) or info.get("TimeLimit.truncated", False)

            if ep_over:
                s = self._sum[i]
                self.logger.record(f"{self.prefix}/E_sum", s["E"])
                self.logger.record(f"{self.prefix}/V_sum", s["V"])
                self.logger.record(f"{self.prefix}/C_sum", s["C"])
                self.logger.record(f"{self.prefix}/penalty_sum", s["P"])
                self.logger.record(f"{self.prefix}/steps_ep", s["steps"])
                self.logger.dump(self.model.num_timesteps)
                self._init_sum(i)

        # periodic interval stats
        if self.interval and (self.num_timesteps % self.interval == 0):
            self._record_interval()
        return True
