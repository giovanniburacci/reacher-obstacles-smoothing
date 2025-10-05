from typing import Optional, Sequence, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import Wrapper
from stable_baselines3.common.callbacks import BaseCallback


# =========================
#  SJRS reward wrapper
# =========================
class SJRSRewardWrapper(Wrapper):
    """
    Slew-Jerk Reward Shaping (SJRS).

    Penalizes three components on a chosen signal `vec`:
      - Energy (E):   ||vec||^2
      - Slew (V):     ||Δvec||^2
      - Jerk (C):     ||Δ²vec||^2

    where vec is either:
      - "torque": physical actuator torques τ read from env (MuJoCo qfrc_actuator)
      - "action": the action provided to env.step(action)

    Behavior is aligned with sjrs_old.py:
      - histories are zero-initialized at reset (Δ, Δ² are non-zero on first step)
      - legacy flat info keys are emitted (sjrs_temporal, sjrs_magnitude, sjrs_curvature, sjrs_penalty)

    Additionally, it exposes new-style fields:
      - E_step, V_step, C_step, penalty_step
    """

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
        assert sjrs_mode in ("torque", "action"), "invalid sjrs_mode"
        self.sjrs_mode = sjrs_mode

        # Determine actuator dimension
        assert hasattr(self.env.action_space, "shape") and self.env.action_space.shape is not None
        self.n_act = int(self.env.action_space.shape[0])

        # Histories (zero-initialized to match sjrs_old behavior)
        self._last_tau = np.zeros(self.n_act, dtype=np.float64)
        self._last_dtau = np.zeros(self.n_act, dtype=np.float64)
        self._last_vec  = np.zeros(self.n_act, dtype=np.float64)
        self._last_dvec = np.zeros(self.n_act, dtype=np.float64)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # zero histories (sjrs_old behavior)
        self._last_tau[:] = 0.0
        self._last_dtau[:] = 0.0
        self._last_vec[:]  = 0.0
        self._last_dvec[:] = 0.0
        return obs, info

    def _read_tau(self) -> np.ndarray:
        # Read actuator torques (MuJoCo). Fall back to zeros if not available.
        try:
            tau = np.asarray(self.env.unwrapped.data.qfrc_actuator, dtype=np.float64)[:self.n_act]
        except Exception:
            tau = np.zeros(self.n_act, dtype=np.float64)
        return tau

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # --- reference torques (always read for legacy/diagnostics) ---
        tau = self._read_tau()
        dtau = tau - self._last_tau
        d2tau = dtau - self._last_dtau

        # --- choose penalized signal vec ---
        if self.sjrs_mode == "torque":
            vec = tau
        else:  # "action"
            vec = np.asarray(action, dtype=np.float64)

        # First and second differences on the chosen vec
        dvec  = vec  - self._last_vec
        d2vec = dvec - self._last_dvec

        # Raw components (new-style names)
        E_step = float(np.dot(vec,  vec))   # energy (||vec||^2)
        V_step = float(np.dot(dvec, dvec))  # slew   (||Δvec||^2)
        C_step = float(np.dot(d2vec,d2vec)) # jerk   (||Δ²vec||^2)

        # Weighted components (old-style names: magnitude/temporal/curvature)
        magnitude_w = self.lambda_a * E_step
        temporal_w  = self.lambda_t * V_step
        curvature_w = self.lambda_c * C_step

        penalty_step = magnitude_w + temporal_w + curvature_w
        reward = float(reward) - float(penalty_step)

        # --- logging (nested sjrs dict) ---
        sjrs_info = info.setdefault("sjrs", {})
        sjrs_info.update({
            # mode numeric (old behavior)
            "mode": {"torque": 0, "action": 1}[self.sjrs_mode],

            # weighted terms (affect reward) — old-style keys
            "magnitude": magnitude_w,
            "temporal":  temporal_w,
            "curvature": curvature_w,
            "penalty":   float(penalty_step),

            # raw wrt chosen signal — old-style keys
            "magnitude_raw": E_step,
            "temporal_raw":  V_step,
            "curvature_raw": C_step,

            # also expose the new-style field names for compatibility
            "E_step": E_step,
            "V_step": V_step,
            "C_step": C_step,
            "penalty_step": float(penalty_step),

            # tiny norms (diagnostics)
            "vec_norm":   float(np.linalg.norm(vec)),
            "dvec_norm":  float(np.linalg.norm(dvec)),
            "d2vec_norm": float(np.linalg.norm(d2vec)),

            # reference τ norms (diagnostics)
            "tau_norm":    float(np.linalg.norm(tau)),
            "dtau_norm":   float(np.linalg.norm(dtau)),
            "d2tau_norm":  float(np.linalg.norm(d2tau)),

            # lambdas
            "lambda_t": float(self.lambda_t),
            "lambda_a": float(self.lambda_a),
            "lambda_c": float(self.lambda_c),

            "active": 1.0,
        })

        # --- legacy flat keys (τ-based), for strict backward-compat ---
        info["sjrs_temporal"]  = float(np.dot(dtau,  dtau))   # τ-based Δ
        info["sjrs_magnitude"] = float(np.dot(tau,   tau))    # τ-based magnitude
        info["sjrs_curvature"] = float(np.dot(d2tau, d2tau))  # τ-based Δ²
        info["sjrs_penalty"]   = float(penalty_step)

        # --- state updates ---
        self._last_dtau = dtau
        self._last_tau  = tau
        self._last_dvec = dvec
        self._last_vec  = vec

        return obs, reward, terminated, truncated, info


# =========================
#  TensorBoard callback
# =========================
class SJRSCallback(BaseCallback):
    """
    Logs both old-style and new-style SJRS metrics.

    Per-episode (per env i):
        sjrs/E_sum, V_sum, C_sum, penalty_sum, steps_ep

    Interval (rolling means + p95):
        sjrs/E_step_mean, V_step_mean, C_step_mean, penalty_step_mean
        sjrs/temporal_raw_p95, magnitude_raw_p95, curvature_raw_p95
        sjrs/temporal_mean, magnitude_mean, curvature_mean, penalty_mean
        sjrs/lambda_a_mean, lambda_t_mean, lambda_c_mean
    """
    def __init__(self, interval: int = 500, prefix: str = "sjrs", verbose: int = 0):
        super().__init__(verbose=verbose)
        self.interval = int(interval)
        self.prefix = prefix

        # canonical keys we capture from info["sjrs"]
        self.keys_weighted = ["temporal", "magnitude", "curvature", "penalty"]
        self.keys_raw = ["temporal_raw", "magnitude_raw", "curvature_raw"]
        self.keys_new = ["E_step", "V_step", "C_step", "penalty_step"]

        # episode accumulators per env
        self._sum: Dict[int, Dict[str, float]] = {}
        # rolling buffers for interval stats
        self.buf: Dict[str, list] = {k: [] for k in (self.keys_weighted + self.keys_raw + self.keys_new + [
            "lambda_a", "lambda_t", "lambda_c"
        ])}

        # map legacy flat keys -> canonical raw keys
        self._legacy_map = {
            "sjrs_temporal": "temporal_raw",
            "sjrs_magnitude": "magnitude_raw",
            "sjrs_curvature": "curvature_raw",
            "sjrs_penalty": "penalty",
        }

    # ---------- SB3 hooks ----------

    def _on_training_start(self) -> None:
        # Heartbeat flag
        self.logger.record(f"{self.prefix}/alive", 1.0)
        self.logger.dump(self.model.num_timesteps)

    def _on_step(self) -> bool:
        infos = self._infos_list()
        dones = self.locals.get("dones", None)

        # capture step metrics
        for info in infos:
            if not isinstance(info, dict):
                continue

            # nested sjrs dict
            sj = info.get("sjrs", None)
            if isinstance(sj, dict):
                # weighted & raw (old-style)
                for k in self.keys_weighted + self.keys_raw:
                    v = sj.get(k, None)
                    if v is not None:
                        self._append(k, v)
                # new-style fields
                for k in self.keys_new:
                    v = sj.get(k, None)
                    if v is not None:
                        self._append(k, v)
                # lambdas if present
                for k in ("lambda_a", "lambda_t", "lambda_c"):
                    v = sj.get(k, None)
                    if v is not None:
                        self._append(k, v)

            # legacy flat keys (τ-based)
            for flat, canon in self._legacy_map.items():
                v = info.get(flat, None)
                if v is not None:
                    self._append(canon, v)

        # episode sums per env (when a done occurs)
        if isinstance(dones, (list, tuple, np.ndarray)):
            for i, d in enumerate(dones):
                if d:
                    self._emit_episode(i)

        # periodic interval flush
        if self.interval > 0 and (self.num_timesteps % self.interval == 0):
            self._flush()

        return True

    def _on_rollout_end(self) -> None:
        self._flush()

    # ---------- helpers ----------

    def _init_sum(self, i: int):
        self._sum[i] = dict(E=0.0, V=0.0, C=0.0, P=0.0, steps=0)

    def _infos_list(self):
        infos = self.locals.get("infos", None)
        if infos is None:
            return []
        return infos if isinstance(infos, (list, tuple)) else [infos]

    def _append(self, k: str, v: float):
        self.buf.setdefault(k, []).append(float(v))

    def _emit_episode(self, i: int):
        # Use interval buffers to compute episode sums if available
        sums = self._sum.get(i)
        if sums is None:
            self._init_sum(i)
            sums = self._sum[i]

        # Try to aggregate last step’s E/V/C/penalty if present in buffers
        for k_src, k_dst in [("E_step", "E"), ("V_step", "V"), ("C_step", "C"), ("penalty_step", "P")]:
            arr = self.buf.get(k_src, [])
            if len(arr) > 0:
                sums[k_dst] += float(arr[-1])
        sums["steps"] += 1

        # Log episode aggregates and reset
        self.logger.record(f"{self.prefix}/E_sum", sums["E"])
        self.logger.record(f"{self.prefix}/V_sum", sums["V"])
        self.logger.record(f"{self.prefix}/C_sum", sums["C"])
        self.logger.record(f"{self.prefix}/penalty_sum", sums["P"])
        self.logger.record(f"{self.prefix}/steps_ep", sums["steps"])
        self.logger.dump(self.model.num_timesteps)
        self._init_sum(i)

    def _flush(self):
        any_logged = False

        # means for new-style raw components
        for k in ("E_step", "V_step", "C_step", "penalty_step"):
            vals = self.buf.get(k, [])
            if vals:
                self.logger.record(f"{self.prefix}/{k}_mean", float(np.mean(vals)))
                any_logged = True

        # means for old-style weighted terms
        for k in ("temporal", "magnitude", "curvature", "penalty"):
            vals = self.buf.get(k, [])
            if vals:
                self.logger.record(f"{self.prefix}/{k}_mean", float(np.mean(vals)))
                any_logged = True

        # p95 for *_raw (spike diagnostics)
        for raw_k in ("temporal_raw", "magnitude_raw", "curvature_raw"):
            vals = self.buf.get(raw_k, [])
            if len(vals) >= 10:
                p95 = float(np.percentile(vals, 95))
                self.logger.record(f"{self.prefix}/{raw_k}_p95", p95)
                any_logged = True

        # average lambdas if present
        for k in ("lambda_a", "lambda_t", "lambda_c"):
            vals = self.buf.get(k, [])
            if vals:
                self.logger.record(f"{self.prefix}/{k}_mean", float(np.mean(vals)))
                any_logged = True

        # heartbeat + flush
        if any_logged:
            self.logger.record(f"{self.prefix}/timesteps", self.num_timesteps)
            self.logger.dump(self.model.num_timesteps)

        # clear buffers
        for k in list(self.buf.keys()):
            self.buf[k].clear()
