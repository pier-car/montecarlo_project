#!/usr/bin/env python3
"""Run a NEPHELE dispersion-assimilation scenario from a YAML config.

Usage:
    python scripts/run_scenario.py --config data/demo/release.yaml
    python scripts/run_scenario.py --config data/demo/release.yaml --serve
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))

from nephele.driver import DispersionDriver, ScenarioConfig  # noqa: E402
from nephele.geo.meteo import MeteoColumn  # noqa: E402
from nephele.service.app import build_hazard_geojson  # noqa: E402


def _load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ImportError:
        # Minimal fallback: accept JSON if PyYAML is unavailable.
        return json.loads(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a NEPHELE scenario")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--horizon", type=float, default=120.0,
                        help="forecast horizon in seconds")
    parser.add_argument("--threshold", type=float, default=1e-6,
                        help="hazard concentration threshold")
    parser.add_argument("--output", type=Path, default=Path("hazard.geojson"))
    parser.add_argument("--serve", action="store_true",
                        help="launch the FastAPI service instead of a one-shot run")
    args = parser.parse_args(argv)

    if args.serve:
        try:
            import uvicorn  # type: ignore

            from nephele.service.app import create_app
        except ImportError:
            print("Install the service extra: pip install 'nephele[service]'",
                  file=sys.stderr)
            return 1
        uvicorn.run(create_app(), host="0.0.0.0", port=8000)  # noqa: S104
        return 0

    cfg = _load_config(args.config)
    column = MeteoColumn(
        wind_speed=float(cfg["wind_speed"]),
        wind_dir_deg=float(cfg["wind_dir_deg"]),
        stability=str(cfg.get("stability", "D")),
    )
    config = ScenarioConfig(
        source_pos=tuple(cfg["source_pos"]),
        source_mass=float(cfg["source_mass"]),
        meteo=column,
        num_particles=int(cfg.get("num_particles", 50000)),
        dt=float(cfg.get("dt", 0.5)),
        decay_rate=float(cfg.get("decay_rate", 0.0)),
    )

    driver = DispersionDriver(config)
    field = driver.forecast(args.horizon)
    geojson = build_hazard_geojson(
        field, config.grid_origin, config.grid_cell, args.threshold
    )
    args.output.write_text(json.dumps(geojson), encoding="utf-8")

    total_mass = float(field.sum() * np.prod(config.grid_cell))
    print(f"Forecast horizon : {args.horizon:.1f} s")
    print(f"Airborne mass    : {total_mass:.4g}")
    print(f"Hazard cells     : {len(geojson['features'])}")
    print(f"Wrote GeoJSON    : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
