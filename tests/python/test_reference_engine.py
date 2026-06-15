import numpy as np
import pytest

from nephele.reference import ReferenceDispersionEngine, UniformMeteo


def test_mass_conserved_without_decay():
    meteo = UniformMeteo(mean_wind=(1.0, 0.0, 0.0), sigma2=(0.3, 0.3, 0.3),
                         epsilon=0.1)
    engine = ReferenceDispersionEngine(
        meteo=meteo, num_particles=20000, dt=0.1,
        source_pos=(5.0, 5.0, 50.0), source_mass=10.0, seed=42,
    )
    engine.advance(5)
    conc = engine.concentration((-100.0, -100.0, -100.0), (2.0, 2.0, 2.0),
                                (150, 150, 150))
    total = conc.sum() * (2.0 * 2.0 * 2.0)
    assert total == pytest.approx(10.0, rel=1e-9)


def test_mean_advection_matches_wind():
    wind = 2.5
    meteo = UniformMeteo(mean_wind=(wind, 0.0, 0.0), sigma2=(0.4, 0.4, 0.4),
                         epsilon=0.1)
    engine = ReferenceDispersionEngine(
        meteo=meteo, num_particles=40000, dt=0.1,
        source_pos=(0.0, 0.0, 50.0), seed=1,
    )
    steps = 100
    engine.advance(steps)
    mean_x = engine.positions[:, 0].mean()
    assert mean_x == pytest.approx(wind * 0.1 * steps, rel=0.05)


def test_ground_reflection_keeps_particles_above_surface():
    meteo = UniformMeteo(mean_wind=(0.0, 0.0, 0.0), sigma2=(1.0, 1.0, 2.0),
                         epsilon=0.5, ground=0.0)
    engine = ReferenceDispersionEngine(
        meteo=meteo, num_particles=20000, dt=0.1,
        source_pos=(0.0, 0.0, 1.0), seed=9,
    )
    engine.advance(50)
    assert np.all(engine.positions[:, 2] >= 0.0)


def test_decay_reduces_total_mass():
    meteo = UniformMeteo()
    engine = ReferenceDispersionEngine(
        meteo=meteo, num_particles=5000, dt=0.5, decay_rate=0.1,
        source_mass=1.0, source_pos=(0.0, 0.0, 50.0), seed=3,
    )
    engine.advance(4)  # t = 2 s
    remaining = engine.mass.sum()
    assert remaining == pytest.approx(np.exp(-0.1 * 2.0), rel=1e-9)


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        ReferenceDispersionEngine(meteo=UniformMeteo(), num_particles=0)
    with pytest.raises(ValueError):
        UniformMeteo(epsilon=-1.0)
