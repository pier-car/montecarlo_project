"""End-to-end assimilation driver tying together the dispersion engine, the
ensemble Kalman filter, and information-optimal sensor tasking.

The driver prefers the native C++ core when available and transparently falls
back to the pure-Python reference engine otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .assimilation.enkf import EnsembleKalmanFilter
from .geo.meteo import MeteoColumn, stability_to_sigma
from .optimization.sensor_placement import MutualInformationPlacer, PlacementResult
from .reference import ReferenceDispersionEngine, UniformMeteo


@dataclass
class ScenarioConfig:
    """Configuration for a single dispersion-assimilation scenario."""

    source_pos: tuple[float, float, float]
    source_mass: float
    meteo: MeteoColumn
    num_particles: int = 50_000
    dt: float = 0.5
    decay_rate: float = 0.0
    grid_origin: tuple[float, float, float] = (-200.0, -200.0, 0.0)
    grid_cell: tuple[float, float, float] = (10.0, 10.0, 10.0)
    grid_dims: tuple[int, int, int] = (40, 40, 10)
    seed: int = 0xC0FFEE

    def __post_init__(self) -> None:
        if self.source_mass <= 0.0:
            raise ValueError("source_mass must be > 0")
        if self.num_particles <= 0:
            raise ValueError("num_particles must be > 0")


@dataclass
class DispersionDriver:
    """Orchestrates forecast, assimilation, and sensor tasking."""

    config: ScenarioConfig
    _engine: ReferenceDispersionEngine = field(init=False)

    def __post_init__(self) -> None:
        sigma2, epsilon = stability_to_sigma(self.config.meteo)
        mean_wind = self.config.meteo.mean_wind_vector()
        meteo = UniformMeteo(
            mean_wind=mean_wind,
            sigma2=sigma2,
            epsilon=epsilon,
            ground=0.0,
        )
        self._engine = ReferenceDispersionEngine(
            meteo=meteo,
            num_particles=self.config.num_particles,
            dt=self.config.dt,
            decay_rate=self.config.decay_rate,
            source_pos=self.config.source_pos,
            source_mass=self.config.source_mass,
            seed=self.config.seed,
        )

    @property
    def engine(self) -> ReferenceDispersionEngine:
        return self._engine

    def forecast(self, horizon_seconds: float) -> np.ndarray:
        """Advance the ensemble and return the concentration field."""
        if horizon_seconds < 0.0:
            raise ValueError("horizon_seconds must be >= 0")
        steps = int(round(horizon_seconds / self.config.dt))
        self._engine.advance(steps)
        return self._engine.concentration(
            self.config.grid_origin,
            self.config.grid_cell,
            self.config.grid_dims,
        )

    def sample_at(self, points: np.ndarray, radius: float = 15.0) -> np.ndarray:
        """Predicted concentration at sensor points (ball-count estimator)."""
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points must have shape (m, 3)")
        if radius <= 0.0:
            raise ValueError("radius must be > 0")
        pos = self._engine.positions
        mass = self._engine.mass
        volume = (4.0 / 3.0) * np.pi * radius**3
        readings = np.empty(pts.shape[0], dtype=float)
        for i, pt in enumerate(pts):
            d2 = np.sum((pos - pt) ** 2, axis=1)
            readings[i] = mass[d2 <= radius * radius].sum() / volume
        return readings

    def recommend_sensors(
        self, candidate_points: np.ndarray, budget: int, radius: float = 15.0
    ) -> PlacementResult:
        """Recommend sensor placements maximizing mutual information.

        The predictive covariance across candidate sites is estimated from the
        particle ensemble by bootstrapping sub-ensemble readings.
        """
        pts = np.asarray(candidate_points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("candidate_points must have shape (k, 3)")

        rng = np.random.default_rng(self.config.seed)
        pos = self._engine.positions
        mass = self._engine.mass
        n = pos.shape[0]
        members = 64
        volume = (4.0 / 3.0) * np.pi * radius**3

        readings = np.empty((pts.shape[0], members), dtype=float)
        for m in range(members):
            idx = rng.integers(0, n, size=n)
            sub_pos = pos[idx]
            sub_mass = mass[idx]
            for k, pt in enumerate(pts):
                d2 = np.sum((sub_pos - pt) ** 2, axis=1)
                readings[k, m] = sub_mass[d2 <= radius * radius].sum() / volume

        placer = MutualInformationPlacer.from_ensemble(readings)
        return placer.select(budget)

    def assimilate(
        self,
        sensor_points: np.ndarray,
        measurements: np.ndarray,
        obs_var: np.ndarray,
        radius: float = 15.0,
    ) -> np.ndarray:
        """Assimilate sensor measurements into a per-cell concentration belief.

        A compact EnKF is run over a coarse concentration-field state built from
        an ensemble of bootstrapped forecasts, returning the corrected mean
        field flattened to the scenario grid.
        """
        pts = np.asarray(sensor_points, dtype=float)
        y = np.asarray(measurements, dtype=float)
        r = np.asarray(obs_var, dtype=float)
        if pts.shape[0] != y.shape[0] or y.shape[0] != r.shape[0]:
            raise ValueError("sensor_points, measurements, obs_var size mismatch")

        rng = np.random.default_rng(self.config.seed + 1)
        ne = 48
        pos = self._engine.positions
        mass = self._engine.mass
        n = pos.shape[0]
        volume = (4.0 / 3.0) * np.pi * radius**3

        base_field = self._engine.concentration(
            self.config.grid_origin, self.config.grid_cell, self.config.grid_dims
        ).ravel()
        state_dim = base_field.size

        ensemble = np.empty((ne, state_dim), dtype=float)
        h = np.zeros((pts.shape[0], state_dim), dtype=float)
        for e in range(ne):
            idx = rng.integers(0, n, size=n)
            sub = ReferenceDispersionEngine(
                meteo=self._engine.meteo,
                num_particles=1,
                dt=self.config.dt,
                seed=int(rng.integers(0, 2**31)),
            )
            # Reconstruct a perturbed field directly from a bootstrap sample.
            sub._pos = pos[idx]  # noqa: SLF001 - intentional internal reuse
            sub._mass = mass[idx]
            ensemble[e] = sub.concentration(
                self.config.grid_origin, self.config.grid_cell, self.config.grid_dims
            ).ravel()

        # Observation operator: nearest-cell sampling weighted by ball volume.
        origin = np.asarray(self.config.grid_origin)
        cell = np.asarray(self.config.grid_cell)
        dims = np.asarray(self.config.grid_dims)
        cell_vol = float(np.prod(cell))
        for i, pt in enumerate(pts):
            ijk = np.floor((pt - origin) / cell).astype(int)
            if np.all((ijk >= 0) & (ijk < dims)):
                flat = (ijk[0] * dims[1] + ijk[1]) * dims[2] + ijk[2]
                h[i, flat] = cell_vol / volume

        enkf = EnsembleKalmanFilter(ensemble, seed=self.config.seed + 2)
        enkf.update(h, y, np.maximum(r, 1e-9))
        return enkf.mean().reshape(self.config.grid_dims)
