from typing import Optional, Sequence, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import Wrapper
from stable_baselines3.common.callbacks import BaseCallback


class SJRSRewardWrapper(Wrapper):
    def __init__(
            self,
            env: gym.Env,
            lambda_t: float = 0.0,
            lambda_a: float = 0.0,
            lambda_c: float = 0.0,
            sjrs_mode: str = "torque",            # "torque" | "action"
    ):
        super().__init__(env)
        self.lambda_t = float(lambda_t)
        self.lambda_a = float(lambda_a)
        self.lambda_c = float(lambda_c)
        assert sjrs_mode in ("torque", "action"), "invalid sjrs_mode"
        self.sjrs_mode = sjrs_mode
        # ---------------

        assert hasattr(self.env.action_space, "shape") and self.env.action_space.shape is not None
        self.n_act = int(self.env.action_space.shape[0])

        # existing state
        self._last_tau = np.zeros(self.n_act, dtype=np.float64)
        self._last_dtau = np.zeros(self.n_act, dtype=np.float64)
        # state for selected vector (τ or u)
        self._last_vec  = np.zeros(self.n_act, dtype=np.float64)
        self._last_dvec = np.zeros(self.n_act, dtype=np.float64)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_tau[:] = 0.0
        self._last_dtau[:] = 0.0
        self._last_vec[:]  = 0.0
        self._last_dvec[:] = 0.0
        # -----------
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # always read physical torques for reference
        tau = np.asarray(self.env.unwrapped.data.qfrc_actuator, dtype=np.float64)[:self.n_act]
        dtau = tau - self._last_tau
        d2tau = dtau - self._last_dtau

        # choose the signal SJRS actually penalizes
        if self.sjrs_mode == "torque":
            vec = tau
        elif self.sjrs_mode == "action":
            vec = np.asarray(action, dtype=np.float64)

        dvec  = vec - self._last_vec
        d2vec = dvec - self._last_dvec

        # raw (λ-independent) terms for the chosen signal
        cost_a = float(np.dot(vec,  vec))     # magnitude
        cost_t = float(np.dot(dvec, dvec))    # temporal (Δ)
        cost_c = float(np.dot(d2vec, d2vec))  # curvature (Δ²)

        # penalty that impacts reward
        penalty = self.lambda_a*cost_a + self.lambda_t*cost_t + self.lambda_c*cost_c
        reward -= penalty
        # ---------------------------------------------------------

        # logging
        sjrs_info = info.setdefault("sjrs", {})
        sjrs_info.update({
            # which signal is used (for TB)
            "mode": {"torque":0, "action":1}[self.sjrs_mode],

            # λ-weighted (affect reward)
            "magnitude": self.lambda_a*cost_a,
            "temporal":  self.lambda_t*cost_t,
            "curvature": self.lambda_c*cost_c,
            "penalty":   penalty,

            # raw wrt chosen signal
            "magnitude_raw": cost_a,
            "temporal_raw":  cost_t,
            "curvature_raw": cost_c,

            # reference (physical torque) norms stay available
            "tau_norm":  float(np.linalg.norm(tau)),
            "dtau_norm": float(np.linalg.norm(dtau)),
            "d2tau_norm":float(np.linalg.norm(d2tau)),

            # λ’s for visibility
            "lambda_t": float(self.lambda_t),
            "lambda_a": float(self.lambda_a),
            "lambda_c": float(self.lambda_c),

            "active": 1.0,
        })

        # legacy flat keys
        info["sjrs_temporal"] = float(np.dot(dtau, dtau))   # still the τ-based Δ term
        info["sjrs_magnitude"] = float(np.dot(tau, tau))    # τ-based magnitude
        info["sjrs_curvature"] = float(np.dot(d2tau, d2tau))# τ-based Δ²
        info["sjrs_penalty"] = penalty

        # --- state updates ---
        self._last_dtau = dtau
        self._last_tau  = tau
        self._last_dvec = dvec
        self._last_vec  = vec

        return obs, reward, terminated, truncated, info


class SJRSCallback(BaseCallback):
    """
    SJRS logger
    """
    def __init__(self, interval: int = 500, prefix: str = "sjrs", verbose: int = 0):
        super().__init__(verbose=verbose)
        self.interval = int(interval)
        self.prefix = prefix

        # keys we expect to log
        self.keys = [
            # λ-weighted terms (impact reward)
            "temporal", "magnitude", "curvature", "penalty",
            # raw diagnostics (no λ)
            "temporal_raw", "magnitude_raw", "curvature_raw",
            # norms & lambdas
            "tau_norm", "dtau_norm", "d2tau_norm",
            "lambda_t", "lambda_a", "lambda_c",
            # misc
            "active",
            "penalty"
        ]
        self.buf: Dict[str, list] = {k: [] for k in self.keys}

        # Mapping from legacy flat keys to canonical names
        self._legacy_map = {
            "sjrs_temporal": "temporal_raw",
            "sjrs_magnitude": "magnitude_raw",
            "sjrs_curvature": "curvature_raw",
            "sjrs_penalty": "penalty",
        }

    # ---------- SB3 hooks ----------
    def _on_training_start(self) -> None:
        # heartbeat
        self.logger.record(f"{self.prefix}/alive", 1.0)
        self.logger.dump(0)

    def _on_step(self) -> bool:
        for info in self._infos_list():
            if not info:
                continue

            # prefer nested dict produced by our wrapper
            c = info.get(self.prefix, None)
            if isinstance(c, dict):
                self._capture_from_dict(c)

            # also capture legacy flat keys if present
            for flat, canon in self._legacy_map.items():
                v = info.get(flat, None)
                if v is not None:
                    self._append(canon, v)

        # periodic emit
        if self.interval > 0 and (self.num_timesteps % self.interval == 0):
            self._flush()
        return True

    def _on_rollout_end(self) -> None:
        self._flush()

    # ---------- helpers ----------
    def _infos_list(self):
        infos = self.locals.get("infos", None)
        if infos is None:
            return []
        # SubprocVecEnv -> list[dict], DummyVecEnv -> dict
        return infos if isinstance(infos, (list, tuple)) else [infos]

    def _capture_from_dict(self, d: Dict[str, Any]) -> None:
        for k in self.keys:
            v = d.get(k, None)
            if v is not None:
                self._append(k, v)

        # accept prefixed keys too (e.g., 'sjrs_temporal_raw')
        for k in list(self.keys):
            pref = f"{self.prefix}_{k}"
            if pref in d:
                self._append(k, d[pref])

    def _append(self, key: str, value: Any) -> None:
        try:
            self.buf[key].append(float(value))
        except Exception:
            pass

    def _flush(self) -> None:
        any_logged = False
        # means
        for k, vals in self.buf.items():
            if not vals:
                continue
            m = float(np.mean(vals))
            self.logger.record(f"{self.prefix}/{k}_mean", m)
            any_logged = True

        # p95 for *_raw to catch rare spikes
        for raw_k in ("temporal_raw", "magnitude_raw", "curvature_raw"):
            vals = self.buf.get(raw_k, [])
            if len(vals) >= 10:
                p95 = float(np.percentile(vals, 95))
                self.logger.record(f"{self.prefix}/{raw_k}_p95", p95)
                any_logged = True

        if any_logged:
            # Force flush so TB updates immediately
            self.logger.dump(self.model.num_timesteps)

        # clear buffers
        for k in self.buf:
            self.buf[k].clear()
