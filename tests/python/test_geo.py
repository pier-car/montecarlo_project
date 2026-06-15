import json
from pathlib import Path

import numpy as np
import pytest

from nephele.geo.canopy import load_building_footprints
from nephele.geo.meteo import MeteoColumn, stability_to_sigma


def test_stability_to_sigma_monotonic_in_instability():
    column_unstable = MeteoColumn(wind_speed=5.0, wind_dir_deg=0.0, stability="A")
    column_stable = MeteoColumn(wind_speed=5.0, wind_dir_deg=0.0, stability="F")
    sig_u, eps_u = stability_to_sigma(column_unstable)
    sig_s, eps_s = stability_to_sigma(column_stable)
    # Unstable conditions => larger velocity variances.
    assert sig_u[2] > sig_s[2]
    assert eps_u == pytest.approx(eps_s)  # epsilon depends on u*, not class here


def test_mean_wind_vector_points_downwind():
    # Wind FROM the west (270 deg) blows TOWARD the east (+x).
    column = MeteoColumn(wind_speed=4.0, wind_dir_deg=270.0, stability="D")
    u, v, w = column.mean_wind_vector()
    assert u > 0.0
    assert abs(v) < 1e-6
    assert w == 0.0


def test_meteo_validation():
    with pytest.raises(ValueError):
        MeteoColumn(wind_speed=-1.0, wind_dir_deg=0.0)
    with pytest.raises(ValueError):
        MeteoColumn(wind_speed=1.0, wind_dir_deg=0.0, stability="Z")


def test_load_building_footprints(tmp_path: Path):
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"height": 20.0},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[0.0, 0.0], [30.0, 0.0], [30.0, 30.0], [0.0, 30.0],
                         [0.0, 0.0]]
                    ],
                },
            }
        ],
    }
    path = tmp_path / "buildings.geojson"
    path.write_text(json.dumps(fc), encoding="utf-8")

    canopy = load_building_footprints(path, cell_size=5.0)
    # Centre of the building must carry its height.
    assert canopy.ground_height(15.0, 15.0) == pytest.approx(20.0)
    # Far outside => no obstacle.
    assert canopy.ground_height(1000.0, 1000.0) == 0.0
    assert canopy.roughness_length() > 0.0


def test_load_building_footprints_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_building_footprints(tmp_path / "nope.geojson", cell_size=5.0)
