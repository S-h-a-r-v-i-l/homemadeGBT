# Tests for oblivious tree fitting and prediction.
import numpy as np

from gbt.tree.oblivious_tree import fit
from gbt.data.binning import assign_bins
from gbt.config import lambda_


def _sample_bins(seed=3, n=80):
    rng = np.random.RandomState(seed)
    f0 = rng.uniform(0, 100, n)
    f1 = rng.uniform(0, 100, n)
    bin0, _ = assign_bins(f0)
    bin1, _ = assign_bins(f1)
    return {0: bin0, 1: bin1}, f0, f1, rng


def test_fit_leaf_values_accuracy():
    all_feature_bins, f0, _, rng = _sample_bins()
    n = len(f0)

    g_array = np.where(f0 < 50, -1.0, 1.0) + rng.normal(0, 0.05, n)
    h_array = np.ones(n)

    splits, leaf_values, node_assignments = fit(g_array, h_array, all_feature_bins)

    assert len(splits) > 0

    num_leaves = len(leaf_values)
    G = np.zeros(num_leaves)
    H = np.zeros(num_leaves)
    np.add.at(G, node_assignments, g_array)
    np.add.at(H, node_assignments, h_array)
    expected_leaf_values = -G / (H + lambda_)

    assert np.allclose(leaf_values, expected_leaf_values, atol=1e-8)


def test_fit_G_and_H_sum_conserved():
    all_feature_bins, f0, _, rng = _sample_bins()
    n = len(f0)

    g_array = np.where(f0 < 50, -1.0, 1.0) + rng.normal(0, 0.05, n)
    h_array = np.ones(n)

    _, leaf_values, node_assignments = fit(g_array, h_array, all_feature_bins)

    num_leaves = len(leaf_values)
    G = np.zeros(num_leaves)
    H = np.zeros(num_leaves)
    np.add.at(G, node_assignments, g_array)
    np.add.at(H, node_assignments, h_array)

    assert np.isclose(G.sum(), g_array.sum(), atol=1e-8)
    assert np.isclose(H.sum(), h_array.sum(), atol=1e-8)


def test_fit_no_split_worth_it():
    all_feature_bins, f0, _, _ = _sample_bins()
    n = len(f0)

    g_array = np.zeros(n)
    h_array = np.ones(n)

    splits, leaf_values, node_assignments = fit(g_array, h_array, all_feature_bins)

    assert splits == []
    assert len(leaf_values) == 1
    assert leaf_values[0] == 0.0
    assert np.all(node_assignments == 0)
