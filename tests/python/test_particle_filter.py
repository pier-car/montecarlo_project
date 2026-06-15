import numpy as np
import pytest

from nephele.assimilation.particle_filter import BootstrapParticleFilter


def test_particle_filter_converges_to_true_source():
    rng = np.random.default_rng(0)
    true_x = 7.0
    particles = rng.uniform(-20.0, 20.0, size=(4000, 1))
    pf = BootstrapParticleFilter(particles, seed=1)

    def loglik(p: np.ndarray) -> np.ndarray:
        return -0.5 * ((p[:, 0] - true_x) / 2.0) ** 2

    for _ in range(5):
        pf.update(loglik)

    estimate = pf.estimate()[0]
    assert estimate == pytest.approx(true_x, abs=1.0)


def test_resampling_restores_effective_sample_size():
    rng = np.random.default_rng(2)
    particles = rng.uniform(-5.0, 5.0, size=(2000, 1))
    pf = BootstrapParticleFilter(particles, seed=3)

    def loglik(p: np.ndarray) -> np.ndarray:
        return -0.5 * (p[:, 0] - 1.0) ** 2

    pf.update(loglik, resample_threshold=0.9)
    # After resampling weights are reset to uniform => ESS ~ N.
    ess = pf.effective_sample_size()
    assert ess > 0.5 * particles.shape[0]


def test_predict_applies_transition():
    particles = np.zeros((100, 1))
    pf = BootstrapParticleFilter(particles, seed=5)

    def transition(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return p + 1.0

    pf.predict(transition)
    assert np.allclose(pf.particles, 1.0)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        BootstrapParticleFilter(np.zeros((1, 1)))
    pf = BootstrapParticleFilter(np.zeros((10, 1)))
    with pytest.raises(ValueError):
        pf.update(lambda p: np.zeros(5))  # wrong length
