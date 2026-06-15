"""Stochastic Ensemble Kalman Filter (perturbed-observations variant).

NumPy reference implementation mirroring ``nephele._core.EnsembleKalmanFilter``.
Used for high-level assimilation orchestration and as a differential-testing
oracle for the C++ implementation.
"""

from __future__ import annotations

import numpy as np


class EnsembleKalmanFilter:
    """Stochastic EnKF over an ensemble of state vectors.

    Parameters
    ----------
    ensemble:
        Array of shape ``(ensemble_size, state_dim)`` holding the forecast
        ensemble. A copy is stored internally.
    seed:
        Seed for the observation-perturbation RNG.
    """

    def __init__(self, ensemble: np.ndarray, seed: int = 0xA11CE) -> None:
        ens = np.asarray(ensemble, dtype=float)
        if ens.ndim != 2:
            raise ValueError("ensemble must be 2-D (ensemble_size, state_dim)")
        if ens.shape[0] < 2:
            raise ValueError("ensemble_size must be >= 2")
        self._ensemble = ens.copy()
        self._rng = np.random.default_rng(seed)

    @property
    def ensemble(self) -> np.ndarray:
        return self._ensemble

    @property
    def ensemble_size(self) -> int:
        return self._ensemble.shape[0]

    @property
    def state_dim(self) -> int:
        return self._ensemble.shape[1]

    def mean(self) -> np.ndarray:
        return self._ensemble.mean(axis=0)

    def update(
        self,
        obs_operator: np.ndarray,
        observations: np.ndarray,
        obs_var: np.ndarray,
    ) -> None:
        """Assimilate ``observations = H x + noise`` and update the ensemble.

        Parameters
        ----------
        obs_operator:
            Observation matrix ``H`` of shape ``(m, state_dim)``.
        observations:
            Measured vector ``y`` of length ``m``.
        obs_var:
            Diagonal observation-error variances ``R`` of length ``m``.
        """
        h = np.asarray(obs_operator, dtype=float)
        y = np.asarray(observations, dtype=float)
        r = np.asarray(obs_var, dtype=float)

        m = y.shape[0]
        if m == 0:
            raise ValueError("no observations provided")
        if h.shape != (m, self.state_dim):
            raise ValueError("obs_operator must have shape (m, state_dim)")
        if r.shape != (m,):
            raise ValueError("obs_var must have shape (m,)")
        if np.any(r <= 0.0):
            raise ValueError("obs_var must be strictly positive")

        ne = self.ensemble_size
        xbar = self.mean()
        anomalies = self._ensemble - xbar  # (ne, n)

        hx = self._ensemble @ h.T  # (ne, m)
        hxbar = xbar @ h.T  # (m,)
        h_anom = hx - hxbar  # (ne, m)

        # Innovation covariance S = HA^T HA / (ne - 1) + R.
        s = (h_anom.T @ h_anom) / (ne - 1) + np.diag(r)
        # Cross covariance C = A^T HA / (ne - 1).
        cross = (anomalies.T @ h_anom) / (ne - 1)  # (n, m)

        # Perturbed observations.
        eta = self._rng.normal(size=(ne, m)) * np.sqrt(r)
        innovation = (y + eta) - hx  # (ne, m)

        # Solve S w^T = innovation^T  ->  w = innovation @ S^{-1}.
        w = np.linalg.solve(s, innovation.T).T  # (ne, m)
        self._ensemble = self._ensemble + w @ cross.T
