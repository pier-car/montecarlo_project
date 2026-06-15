"""Pure-Python reference Lagrangian stochastic dispersion engine.

This is a NumPy-vectorised mirror of the C++ compute core. It is intentionally
simple and readable: it serves as the executable specification used to validate
the optimised C++/CUDA implementation via differential testing, and lets the
Python package run end-to-end without the compiled extension.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class UniformMeteo:
    """Homogeneous, stationary Gaussian turbulence field."""

    mean_wind: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sigma2: tuple[float, float, float] = (0.5, 0.5, 0.5)
    epsilon: float = 0.1
    ground: float = 0.0

    def __post_init__(self) -> None:
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0")
        if any(s <= 0.0 for s in self.sigma2):
            raise ValueError("sigma2 components must be > 0")


@dataclass
class ReferenceDispersionEngine:
    """Reference Lagrangian stochastic dispersion engine.

    Integrates the Thomson (1987) well-mixed generalized Langevin equation for a
    uniform turbulence field. Mean concentration is recovered by histogram
    binning of the particle ensemble.
    """

    meteo: UniformMeteo
    num_particles: int = 100_000
    dt: float = 0.1
    c0: float = 4.0
    decay_rate: float = 0.0
    source_pos: tuple[float, float, float] = (0.0, 0.0, 1.0)
    source_mass: float = 1.0
    seed: int = 0xC0FFEE

    _time: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if self.num_particles <= 0:
            raise ValueError("num_particles must be > 0")
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0")
        if self.c0 <= 0.0:
            raise ValueError("c0 must be > 0")
        if self.decay_rate < 0.0:
            raise ValueError("decay_rate must be >= 0")

        self._rng = np.random.default_rng(self.seed)
        n = self.num_particles
        self._pos = np.tile(np.asarray(self.source_pos, dtype=float), (n, 1))
        sigma = np.sqrt(np.asarray(self.meteo.sigma2, dtype=float))
        self._vel = self._rng.normal(size=(n, 3)) * sigma
        self._mass = np.full(n, self.source_mass / n, dtype=float)

    @property
    def time(self) -> float:
        return self._time

    def __len__(self) -> int:
        return self.num_particles

    @property
    def positions(self) -> np.ndarray:
        return self._pos

    @property
    def mass(self) -> np.ndarray:
        return self._mass

    def step(self) -> None:
        m = self.meteo
        sigma2 = np.asarray(m.sigma2, dtype=float)
        mean_wind = np.asarray(m.mean_wind, dtype=float)

        t_l = 2.0 * sigma2 / (self.c0 * m.epsilon)  # Lagrangian timescale
        b = np.sqrt(self.c0 * m.epsilon)

        # Ornstein-Uhlenbeck relaxation (uniform field: no inhomogeneity drift).
        drift = -self._vel / t_l
        noise = self._rng.normal(size=self._vel.shape) * (b * np.sqrt(self.dt))
        self._vel = self._vel + drift * self.dt + noise

        self._pos = self._pos + (mean_wind + self._vel) * self.dt

        # Reflecting ground boundary.
        below = self._pos[:, 2] < m.ground
        self._pos[below, 2] = 2.0 * m.ground - self._pos[below, 2]
        self._vel[below, 2] = -self._vel[below, 2]

        self._mass *= np.exp(-self.decay_rate * self.dt)
        self._time += self.dt

    def advance(self, n: int) -> None:
        for _ in range(n):
            self.step()

    def concentration(
        self,
        origin: tuple[float, float, float],
        cell_size: tuple[float, float, float],
        dims: tuple[int, int, int],
    ) -> np.ndarray:
        """Bin particle mass into a regular grid, returning concentration."""
        origin = np.asarray(origin, dtype=float)
        cell = np.asarray(cell_size, dtype=float)
        dims_arr = np.asarray(dims, dtype=int)
        if np.any(cell <= 0.0):
            raise ValueError("cell_size must be > 0")
        if np.any(dims_arr <= 0):
            raise ValueError("dims must be > 0")

        idx = np.floor((self._pos - origin) / cell).astype(int)
        inside = np.all((idx >= 0) & (idx < dims_arr), axis=1)
        idx = idx[inside]
        mass = self._mass[inside]

        grid = np.zeros(tuple(int(d) for d in dims_arr), dtype=float)
        np.add.at(grid, (idx[:, 0], idx[:, 1], idx[:, 2]), mass)
        cell_volume = float(np.prod(cell))
        return grid / cell_volume
