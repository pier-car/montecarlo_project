"""NEPHELE — Networked Ensemble Plume-Hazard Estimation & Lagrangian Engine.

High-level Python orchestration for the NEPHELE CBRN dispersion stack. The
heavy compute lives in the C++ core (``nephele._core``), exposed via pybind11.
A pure-Python reference engine (:mod:`nephele.reference`) is always available so
the package is usable and testable without the compiled extension.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .assimilation.enkf import EnsembleKalmanFilter
from .assimilation.particle_filter import BootstrapParticleFilter
from .optimization.sensor_placement import MutualInformationPlacer, PlacementResult
from .reference import ReferenceDispersionEngine, UniformMeteo

try:  # pragma: no cover - exercised only when the extension is built
    from . import _core  # type: ignore
    _HAS_CORE = True
except ImportError:  # pragma: no cover
    _core = None  # type: ignore
    _HAS_CORE = False

__all__ = [
    "__version__",
    "EnsembleKalmanFilter",
    "BootstrapParticleFilter",
    "MutualInformationPlacer",
    "PlacementResult",
    "ReferenceDispersionEngine",
    "UniformMeteo",
    "has_native_core",
]


def has_native_core() -> bool:
    """Return True if the compiled C++ core (``nephele._core``) is importable."""
    return _HAS_CORE
