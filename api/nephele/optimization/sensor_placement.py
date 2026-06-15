"""Information-optimal mobile-sensor placement via lazy-greedy submodular
maximization of mutual information over an ensemble belief."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlacementResult:
    """Outcome of a sensor-placement optimization."""

    selected: list[int]
    objective: float


class MutualInformationPlacer:
    """Select sensor sites maximizing ``I(A; V\\A)`` under a Gaussian belief.

    The belief covariance is estimated from a forecast ensemble (e.g. the EnKF
    particle cloud), so placement adapts to the live plume estimate. The mutual
    information objective is monotone submodular, so the lazy-greedy algorithm
    attains the ``(1 - 1/e)`` approximation guarantee.
    """

    def __init__(self, covariance: np.ndarray, jitter: float = 1e-9) -> None:
        cov = np.asarray(covariance, dtype=float)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("covariance must be a square matrix")
        if not np.allclose(cov, cov.T, atol=1e-8):
            raise ValueError("covariance must be symmetric")
        self._cov = cov + jitter * np.eye(cov.shape[0])
        self._n = cov.shape[0]
        self._jitter = jitter

    @classmethod
    def from_ensemble(
        cls, ensemble: np.ndarray, **kwargs: float
    ) -> MutualInformationPlacer:
        """Build from an ``(n_sites, n_members)`` ensemble of predicted readings."""
        ens = np.asarray(ensemble, dtype=float)
        if ens.ndim != 2 or ens.shape[1] < 2:
            raise ValueError("ensemble must be (n_sites, n_members>=2)")
        return cls(np.cov(ens), **kwargs)

    def _cond_variance(self, i: int, conditioning: list[int]) -> float:
        """``Var(y_i | y_conditioning)`` via the Schur complement."""
        if not conditioning:
            return float(self._cov[i, i])
        idx = np.fromiter(conditioning, dtype=int)
        sigma_bb = self._cov[np.ix_(idx, idx)]
        sigma_ib = self._cov[i, idx]
        try:
            solved = np.linalg.solve(sigma_bb, sigma_ib)
        except np.linalg.LinAlgError:
            solved = np.linalg.lstsq(sigma_bb, sigma_ib, rcond=None)[0]
        var = self._cov[i, i] - sigma_ib @ solved
        return float(max(var, self._jitter))

    def _gain(self, i: int, selected: list[int]) -> float:
        """Marginal MI gain of adding site ``i`` (Krause et al., 2008)."""
        rest = [j for j in range(self._n) if j != i and j not in selected]
        var_given_a = self._cond_variance(i, selected)
        var_given_rest = self._cond_variance(i, rest)
        return 0.5 * float(np.log(var_given_a / var_given_rest))

    def select(
        self, budget: int, candidates: list[int] | None = None
    ) -> PlacementResult:
        """Lazy-greedy selection of up to ``budget`` sites.

        Returns the selected indices and the accumulated MI objective. Exploits
        submodularity (Minoux's lazy evaluation) for near-linear practical cost,
        retaining the ``(1 - 1/e)`` approximation guarantee.
        """
        if budget <= 0:
            raise ValueError("budget must be a positive integer")
        pool = list(range(self._n)) if candidates is None else list(candidates)
        if any(c < 0 or c >= self._n for c in pool):
            raise ValueError("candidate index out of range")
        budget = min(budget, len(pool))

        selected: list[int] = []
        objective = 0.0

        # Max-heap of (-upper_bound_gain, site, last_evaluated_size).
        heap: list[tuple[float, int, int]] = [
            (-self._gain(i, selected), i, 0) for i in pool
        ]
        heapq.heapify(heap)

        while heap and len(selected) < budget:
            neg_gain, site, evaluated_at = heapq.heappop(heap)
            if evaluated_at == len(selected):
                selected.append(site)
                objective += -neg_gain
            else:
                fresh = self._gain(site, selected)
                heapq.heappush(heap, (-fresh, site, len(selected)))

        return PlacementResult(selected=selected, objective=objective)
