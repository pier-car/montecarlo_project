import numpy as np
import pytest

from nephele.driver import DispersionDriver, ScenarioConfig
from nephele.geo.meteo import MeteoColumn
from nephele.service.app import build_hazard_geojson


def _driver() -> DispersionDriver:
    column = MeteoColumn(wind_speed=3.0, wind_dir_deg=270.0, stability="D")
    config = ScenarioConfig(
        source_pos=(0.0, 0.0, 2.0),
        source_mass=5.0,
        meteo=column,
        num_particles=8000,
        dt=0.5,
        grid_origin=(-200.0, -200.0, 0.0),
        grid_cell=(10.0, 10.0, 10.0),
        grid_dims=(40, 40, 8),
        seed=11,
    )
    return DispersionDriver(config)


def test_forecast_returns_grid_with_positive_mass():
    driver = _driver()
    field = driver.forecast(30.0)
    assert field.shape == (40, 40, 8)
    assert field.sum() > 0.0


def test_recommend_sensors_returns_budget_sites():
    driver = _driver()
    driver.forecast(20.0)
    candidates = np.array(
        [[x, 0.0, 2.0] for x in np.linspace(0.0, 120.0, 8)], dtype=float
    )
    result = driver.recommend_sensors(candidates, budget=3)
    assert len(result.selected) == 3
    assert len(set(result.selected)) == 3


def test_assimilate_returns_corrected_field():
    driver = _driver()
    driver.forecast(20.0)
    sensors = np.array([[30.0, 0.0, 2.0], [60.0, 0.0, 2.0]], dtype=float)
    truth = driver.sample_at(sensors)
    field = driver.assimilate(sensors, truth, np.array([1e-12, 1e-12]))
    assert field.shape == (40, 40, 8)
    assert np.all(np.isfinite(field))


def test_build_hazard_geojson_thresholds_cells():
    conc = np.zeros((3, 3, 2))
    conc[1, 1, 0] = 10.0
    conc[0, 0, 1] = 0.5
    fc = build_hazard_geojson(conc, (0.0, 0.0, 0.0), (10.0, 10.0, 10.0),
                              threshold=1.0)
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 1
    feat = fc["features"][0]
    assert feat["properties"]["concentration"] == pytest.approx(10.0)
    assert feat["geometry"]["type"] == "Polygon"


def test_build_hazard_geojson_validates_input():
    with pytest.raises(ValueError):
        build_hazard_geojson(np.zeros((3, 3)), (0, 0, 0), (1, 1, 1), 1.0)
    with pytest.raises(ValueError):
        build_hazard_geojson(np.zeros((3, 3, 2)), (0, 0, 0), (1, 1, 1), -1.0)
