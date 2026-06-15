import numpy as np
import pytest

from nephele.assimilation.enkf import EnsembleKalmanFilter


def test_enkf_scalar_update_moves_mean_toward_observation():
    rng = np.random.default_rng(0)
    ensemble = rng.normal(loc=0.0, scale=1.0, size=(2000, 1))
    enkf = EnsembleKalmanFilter(ensemble, seed=1)

    prior = enkf.mean()[0]
    enkf.update(np.array([[1.0]]), np.array([5.0]), np.array([0.25]))
    post = enkf.mean()[0]

    assert post > prior
    assert post < 5.0
    assert post > 2.0  # meaningful correction since R < prior variance


def test_enkf_multivariate_reduces_spread():
    rng = np.random.default_rng(2)
    ensemble = rng.normal(size=(3000, 2)) @ np.array([[1.0, 0.4], [0.0, 1.0]])
    enkf = EnsembleKalmanFilter(ensemble, seed=3)

    prior_var = enkf.ensemble.var(axis=0)
    h = np.array([[1.0, 0.0]])  # observe first component
    enkf.update(h, np.array([0.0]), np.array([0.1]))
    post_var = enkf.ensemble.var(axis=0)

    # Observed component variance must shrink.
    assert post_var[0] < prior_var[0]


def test_enkf_rejects_bad_shapes():
    enkf = EnsembleKalmanFilter(np.zeros((10, 2)))
    with pytest.raises(ValueError):
        enkf.update(np.zeros((1, 3)), np.zeros(1), np.ones(1))
    with pytest.raises(ValueError):
        enkf.update(np.zeros((1, 2)), np.zeros(1), np.array([-1.0]))


def test_enkf_rejects_small_ensemble():
    with pytest.raises(ValueError):
        EnsembleKalmanFilter(np.zeros((1, 2)))
