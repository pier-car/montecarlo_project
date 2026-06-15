"""Service subpackage: REST API exposing hazard forecasts and sensor tasking."""

from .app import build_hazard_geojson, create_app

__all__ = ["create_app", "build_hazard_geojson"]
