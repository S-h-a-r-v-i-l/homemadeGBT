# Tests for gbt.tree.histogram: build_feature_histograms and update_node_assignments.
import numpy as np

from gbt.tree.histogram import build_feature_histograms, update_node_assignments
from gbt.config import bin_count


def test_build_feature_histograms_conserves_gradient_and_hessian():
    rng = np.random.RandomState(1)
    n = 37
    g_array = rng.normal(size=n)
    h_array = np.abs(rng.normal(size=n)) + 0.1
    node_assignments = rng.randint(0, 2, size=n).astype(np.int32)
    feature_bins = rng.randint(0, bin_count, size=n).astype(np.int64)

    histogram = build_feature_histograms(g_array, h_array, node_assignments, feature_bins, 2)

    assert np.isclose(histogram[:, :, 0].sum(), g_array.sum(), atol=1e-8)
    assert np.isclose(histogram[:, :, 1].sum(), h_array.sum(), atol=1e-8)


def test_build_feature_histograms_matches_independent_aggregation():
    rng = np.random.RandomState(2)
    n = 50
    num_nodes = 4
    g_array = rng.normal(size=n)
    h_array = np.abs(rng.normal(size=n)) + 0.1
    node_assignments = rng.randint(0, num_nodes, size=n).astype(np.int32)
    feature_bins = rng.randint(0, bin_count, size=n).astype(np.int64)

    histogram = build_feature_histograms(g_array, h_array, node_assignments, feature_bins, num_nodes)

    expected = np.zeros((num_nodes, bin_count, 2), dtype=np.float64)
    np.add.at(expected[:, :, 0], (node_assignments, feature_bins), g_array)
    np.add.at(expected[:, :, 1], (node_assignments, feature_bins), h_array)

    assert np.allclose(histogram, expected, atol=1e-8)


def test_update_node_assignments_sample_dataset():
    feature_bins = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    node_assignments = np.zeros(6, dtype=np.int32)

    result = update_node_assignments(2, feature_bins, node_assignments)

    assert np.array_equal(result, np.array([0, 0, 0, 1, 1, 1]))


def test_update_node_assignments_length_one():
    feature_bins = np.array([3], dtype=np.int64)
    node_assignments = np.zeros(1, dtype=np.int32)

    result = update_node_assignments(2, feature_bins, node_assignments)

    assert np.array_equal(result, np.array([1]))


def test_update_node_assignments_length_zero():
    feature_bins = np.array([], dtype=np.int64)
    node_assignments = np.zeros(0, dtype=np.int32)

    result = update_node_assignments(2, feature_bins, node_assignments)

    assert result.shape == (0,)
