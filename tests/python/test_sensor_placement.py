import numpy as np
import pytest

from nephele.optimization.sensor_placement import MutualInformationPlacer


def _block_covariance() -> np.ndarray:
    # Two tightly-correlated clusters that are weakly correlated across clusters.
    # An optimal placement should pick one site from each cluster.
    base = np.array(
        [
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.9],
            [0.1, 0.1, 0.9, 1.0],
        ]
    )
    return base


def test_placement_selects_one_per_cluster():
    placer = MutualInformationPlacer(_block_covariance())
    result = placer.select(budget=2)
    assert len(result.selected) == 2
    # One index from {0,1} and one from {2,3}.
    cluster_a = {0, 1}
    cluster_b = {2, 3}
    sel = set(result.selected)
    assert len(sel & cluster_a) == 1
    assert len(sel & cluster_b) == 1


def test_placement_objective_is_nonnegative_and_monotone():
    cov = _block_covariance()
    placer = MutualInformationPlacer(cov)
    obj1 = placer.select(budget=1).objective
    obj2 = placer.select(budget=2).objective
    assert obj1 >= 0.0
    assert obj2 >= obj1  # submodular monotone gains


def test_from_ensemble_builds_consistent_placer():
    rng = np.random.default_rng(7)
    latent = rng.normal(size=(2, 200))
    sites = np.vstack([latent[0], latent[0] + 0.01 * rng.normal(size=200),
                       latent[1], latent[1] + 0.01 * rng.normal(size=200)])
    placer = MutualInformationPlacer.from_ensemble(sites)
    result = placer.select(budget=2)
    assert len(result.selected) == 2


def test_placement_validates_budget_and_candidates():
    placer = MutualInformationPlacer(_block_covariance())
    with pytest.raises(ValueError):
        placer.select(budget=0)
    with pytest.raises(ValueError):
        placer.select(budget=1, candidates=[99])


def test_placement_rejects_non_square_covariance():
    with pytest.raises(ValueError):
        MutualInformationPlacer(np.zeros((2, 3)))
