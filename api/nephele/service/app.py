"""FastAPI service exposing live hazard polygons and sensor tasking.

FastAPI is an optional dependency. ``build_hazard_geojson`` is dependency-free
and unit-tested directly; ``create_app`` raises a clear error if FastAPI is not
installed.
"""

from __future__ import annotations

import numpy as np

from ..driver import DispersionDriver, ScenarioConfig
from ..geo.meteo import MeteoColumn


def build_hazard_geojson(
    concentration: np.ndarray,
    origin: tuple[float, float, float],
    cell_size: tuple[float, float, float],
    threshold: float,
) -> dict:
    """Build a GeoJSON FeatureCollection of cells exceeding a hazard threshold.

    The 3-D concentration field is reduced to a 2-D ground footprint by taking
    the column maximum, then thresholded into square cell polygons.
    """
    conc = np.asarray(concentration, dtype=float)
    if conc.ndim != 3:
        raise ValueError("concentration must be a 3-D array (nx, ny, nz)")
    if threshold <= 0.0:
        raise ValueError("threshold must be > 0")

    footprint = conc.max(axis=2)  # column-max over z
    nx, ny = footprint.shape
    dx, dy, _ = cell_size

    features: list[dict] = []
    for ix in range(nx):
        for iy in range(ny):
            value = float(footprint[ix, iy])
            if value < threshold:
                continue
            x0 = origin[0] + ix * dx
            y0 = origin[1] + iy * dy
            x1 = x0 + dx
            y1 = y0 + dy
            features.append(
                {
                    "type": "Feature",
                    "properties": {"concentration": value},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
                        ],
                    },
                }
            )
    return {"type": "FeatureCollection", "features": features}


def create_app(default_scenario: ScenarioConfig | None = None):  # noqa: ANN201
    """Create the FastAPI application.

    Raises
    ------
    RuntimeError
        If FastAPI is not installed.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "FastAPI is required for the service layer. "
            "Install with: pip install 'nephele[service]'"
        ) from exc

    app = FastAPI(title="NEPHELE Hazard Service", version="0.1.0")

    class ForecastRequest(BaseModel):
        source_pos: tuple[float, float, float]
        source_mass: float = Field(gt=0.0)
        wind_speed: float = Field(ge=0.0)
        wind_dir_deg: float
        stability: str = "D"
        horizon_seconds: float = Field(ge=0.0, default=120.0)
        threshold: float = Field(gt=0.0, default=1e-6)

    @app.get("/health")
    def health() -> dict:  # noqa: ANN202
        return {"status": "ok"}

    @app.post("/hazard")
    def hazard(req: ForecastRequest) -> dict:  # noqa: ANN202
        try:
            column = MeteoColumn(
                wind_speed=req.wind_speed,
                wind_dir_deg=req.wind_dir_deg,
                stability=req.stability,
            )
            config = ScenarioConfig(
                source_pos=req.source_pos,
                source_mass=req.source_mass,
                meteo=column,
            )
            driver = DispersionDriver(config)
            field = driver.forecast(req.horizon_seconds)
        except (ValueError, FloatingPointError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return build_hazard_geojson(
            field, config.grid_origin, config.grid_cell, req.threshold
        )

    return app
