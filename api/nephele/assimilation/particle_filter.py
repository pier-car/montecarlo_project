"""Bootstrap (sequential importance resampling) particle filter.

Tracks a non-Gaussian belief over a low-dimensional source/parameter state
(e.g. release location, rate, wind bias) by reweighting and resampling a swarm
of hypotheses against live sensor counts. Complements the EnKF when the
posterior is strongly non-Gaussian.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


class BootstrapParticleFilter:
    """Bootstrap particle filter with systematic resampling.

    Parameters
    ----------
    particles:
        Array of shape ``(num_particles, state_dim)``.
    seed:
        RNG seed for resampling and propagation noise.
    """

    def __init__(self, particles: np.ndarray, seed: int = 0xBEEF) -> None:
        p = np.asarray(particles, dtype=float)
        if p.ndim != 2:
            raise ValueError("particles must be 2-D (num_particles, state_dim)")
        if p.shape[0] < 2:
            raise ValueError("need at least 2 particles")
        self._particles = p.copy()
        self._weights = np.full(p.shape[0], 1.0 / p.shape[0])
        self._rng = np.random.default_rng(seed)

    @property
    def particles(self) -> np.ndarray:
        return self._particles

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    def estimate(self) -> np.ndarray:
        """Weighted-mean state estimate."""
        return np.average(self._particles, axis=0, weights=self._weights)

    def effective_sample_size(self) -> float:
        """Kish effective sample size; low values indicate weight collapse."""
        return float(1.0 / np.sum(self._weights**2))

    def predict(
        self,
        transition: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    ) -> None:
        """Propagate particles through a (possibly stochastic) transition model."""
        self._particles = transition(self._particles, self._rng)
        if self._particles.shape[0] != self._weights.shape[0]:
            raise ValueError("transition must preserve the particle count")

    def update(
        self,
        log_likelihood: Callable[[np.ndarray], np.ndarray],
        resample_threshold: float = 0.5,
    ) -> None:
        """Reweight against an observation and adaptively resample.

        Parameters
        ----------
        log_likelihood:
            Maps the particle array to a vector of log-likelihoods.
        resample_threshold:
            Resample when ESS drops below ``threshold * num_particles``.
        """
        logl = np.asarray(log_likelihood(self._particles), dtype=float)
        if logl.shape != self._weights.shape:
            raise ValueError("log_likelihood must return one value per particle")

        log_w = np.log(self._weights + 1e-300) + logl
        log_w -= log_w.max()  # numerical stabilisation
        w = np.exp(log_w)
        total = w.sum()
        if not np.isfinite(total) or total <= 0.0:
            raise FloatingPointError("particle weights collapsed to zero")
        self._weights = w / total

        n = self._particles.shape[0]
        if self.effective_sample_size() < resample_threshold * n:
            self._resample()

    def _resample(self) -> None:
        """Systematic resampling — low variance, O(n)."""
        n = self._particles.shape[0]
        positions = (self._rng.random() + np.arange(n)) / n
        cumulative = np.cumsum(self._weights)
        cumulative[-1] = 1.0  # guard against rounding
        indices = np.searchsorted(cumulative, positions)
        self._particles = self._particles[indices]
        self._weights = np.full(n, 1.0 / n)
