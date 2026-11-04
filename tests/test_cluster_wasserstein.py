import numpy as np
import pytest
from scipy.stats import wasserstein_distance
from cnr.cluster_wasserstein import cluster_w1, nearest_motion


def test_cluster_bootstrap_matches_explicit_repeated_vehicles():
    a = np.array([[0, 0, 2, 4], [1, 2, 4, 9], [-2, 0, 1, 2.]])
    b = np.array([[1, 2], [2, 8], [-1, 2.]])
    counts = np.array([[1, 1, 1], [3, 0, 0], [0, 1, 2], [2, 1, 0]])
    expected = [wasserstein_distance(np.repeat(a, c, axis=0).ravel(),
                                    np.repeat(b, c, axis=0).ravel()) for c in counts]
    np.testing.assert_allclose(cluster_w1(a, b, counts, batch_size=2), expected, atol=1e-12)


def test_cluster_constant_support_and_invalid_counts():
    np.testing.assert_array_equal(cluster_w1(np.ones((2, 4)), np.ones((2, 1)), [[1, 1]]), [0])
    with pytest.raises(ValueError):
        cluster_w1(np.ones((2, 4)), np.ones((2, 1)), [[0, 0]])


def test_motion_copy_control_and_class_restriction():
    ref = np.array([[0., 1, 2], [2, 1, 0], [0, 1, 2]])
    d, ix = nearest_motion(ref[[0, 1]], ref, np.array([2, 1]), np.array([1, 1, 2]))
    np.testing.assert_array_equal(ix, [2, 1])
    np.testing.assert_array_equal(d, [0, 0])
