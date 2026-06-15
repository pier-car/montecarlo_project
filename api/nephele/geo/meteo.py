"""Meteorological pre-processing: Pasquill stability to turbulence statistics.

Maps surface wind and Pasquill-Gifford stability class to the velocity-variance
and dissipation parameters consumed by the Lagrangian engine. This is a compact,
well-documented closure suitable for tactical use when full mesoscale fields are
unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Empirical ratios sigma_i / u* for the three velocity components in the
# atmospheric surface layer (near-neutral; e.g. Stull 1988).
_SIGMA_OVER_USTAR = (2.5, 2.0, 1.3)

# Coarse mapping of Pasquill class to a turbulence-intensity multiplier.
_STABILITY_INTENSITY = {
    "A": 1.6,  # very unstable
    "B": 1.4,
    "C": 1.2,
    "D": 1.0,  # neutral
    "E": 0.8,
    "F": 0.6,  # very stable
}

_VON_KARMAN = 0.4


@dataclass(frozen=True)
class MeteoColumn:
    """Single-column meteorological state used to drive a uniform field."""

    wind_speed: float
    wind_dir_deg: float  # meteorological convention: direction wind comes FROM
    stability: str = "D"
    roughness_length: float = 0.1

    def __post_init__(self) -> None:
        if self.wind_speed < 0.0:
            raise ValueError("wind_speed must be >= 0")
        if self.stability.upper() not in _STABILITY_INTENSITY:
            raise ValueError(f"unknown Pasquill class: {self.stability}")
        if self.roughness_length <= 0.0:
            raise ValueError("roughness_length must be > 0")

    def mean_wind_vector(self, ref_height: float = 10.0) -> tuple[float, float, float]:
        """Horizontal mean-wind vector (u, v, w) at ``ref_height`` [m/s].

        Direction follows the meteorological convention (FROM), converted to a
        vector pointing in the direction the wind blows TOWARD.
        """
        if ref_height <= 0.0:
            raise ValueError("ref_height must be > 0")
        # Logarithmic profile scaling from the 10 m reference.
        scale = np.log(ref_height / self.roughness_length) / np.log(
            10.0 / self.roughness_length
        )
        speed = self.wind_speed * scale
        bearing = np.deg2rad((self.wind_dir_deg + 180.0) % 360.0)
        u = speed * np.sin(bearing)
        v = speed * np.cos(bearing)
        return (float(u), float(v), 0.0)

    def friction_velocity(self, ref_height: float = 10.0) -> float:
        """Surface friction velocity u* from the log-law (neutral)."""
        return (
            _VON_KARMAN
            * self.wind_speed
            / np.log(ref_height / self.roughness_length)
        )


def stability_to_sigma(
    column: MeteoColumn, ref_height: float = 10.0
) -> tuple[tuple[float, float, float], float]:
    """Return ``(sigma2_xyz, epsilon)`` for a meteorological column.

    Parameters
    ----------
    column:
        The meteorological state.
    ref_height:
        Reference height for u* estimation.

    Returns
    -------
    sigma2:
        Velocity variances ``(sigma_u^2, sigma_v^2, sigma_w^2)`` [m^2/s^2].
    epsilon:
        TKE dissipation rate [m^2/s^3].
    """
    ustar = column.friction_velocity(ref_height)
    intensity = _STABILITY_INTENSITY[column.stability.upper()]
    sigma = tuple((r * ustar * intensity) for r in _SIGMA_OVER_USTAR)
    sigma2 = tuple(max(s * s, 1e-4) for s in sigma)

    # Neutral surface-layer dissipation: epsilon = u*^3 / (kappa * z).
    epsilon = max((ustar**3) / (_VON_KARMAN * ref_height), 1e-4)
    return sigma2, float(epsilon)  # type: ignore[return-value]
