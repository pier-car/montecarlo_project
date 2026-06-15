# NEPHELE — Architecture

NEPHELE is a two-language system with a hard performance boundary: everything
inside the real-time loop is C++; everything that configures, ingests, or
interprets is Python/Julia.

## Component map

```
                         ┌──────────────────────────────────────────┐
   QGIS / GDAL geodata   │              PYTHON / JULIA               │
   (DEM, footprints) ───►│  geo ingest │ assimilation │ optimization │
   met data              └───────┬───────────────┬──────────────┬────┘
                                 │ MeteoField     │ sensor counts│ tasking
                                 ▼                ▼              ▼
                         ┌──────────────────────────────────────────┐
                         │            C++17 COMPUTE CORE             │
                         │  Langevin integrator (OpenMP / CUDA)     │
                         │  concentration grid │ EnKF gain          │
                         └───────────────────┬──────────────────────┘
                                             │ C(x,t), hazard polygons
                                             ▼
                              FastAPI service ► GeoJSON / QGIS
```

## Role separation

| Layer | Language | Responsibility |
|---|---|---|
| Compute core | C++17 (OpenMP, optional CUDA) | Particle advection, grid binning, EnKF gain, reflecting boundaries. Zero per-step heap allocation. |
| Bindings | pybind11 | Zero-copy NumPy ↔ C++ buffer exchange (`nephele._core`). |
| Orchestration | Python 3.11 | Config, QGIS/GDAL ingest, met pre-processing, assimilation driver, sensor optimizer, REST service. |
| Verification | Julia | Reference SDE integrator + well-mixed oracle for differential testing. |
| Presentation | Python (FastAPI) + GeoJSON/QGIS | Hazard polygons, sensor tasking, confidence dashboards. |

## Data flow

1. Geospatial pre-processor turns QGIS building footprints and meteorology into
   a `MeteoField` (mean wind, velocity variances, dissipation, ground height).
2. The driver constructs a dispersion engine and advances the particle ensemble.
3. Live sensor counts feed the EnKF / particle filter, correcting the forecast.
4. The sensor-placement optimizer recommends where to move mobile detectors.
5. The service publishes hazard polygons and tasking as GeoJSON.

## Native vs. reference engine

The Python package ships a NumPy reference engine (`nephele.reference`) that is
always available. When the compiled extension `nephele._core` is built, the same
API is backed by the optimised C++ core. `nephele.has_native_core()` reports
which path is active.
