# Tests for the end-to-end boosting training loop and inference.
import numpy as np

from gbt.ensemble.booster import fit_ensemble
from gbt.ensemble.predictor import predict
from gbt.data.binning import assign_bins
from gbt.loss.logcosh import compute_gradients_and_hessians
from gbt.tree.oblivious_tree import fit
from gbt.config import learning_rate


def _make_data(seed, scale, n=60):
    rng = np.random.RandomState(seed)
    f0 = rng.uniform(0, 100, n)
    f1 = rng.uniform(0, 100, n)
    shape = (f0 - 50) * 0.05 + (f1 - 50) * 0.03 + rng.normal(0, 0.3, n)
    targets = shape * scale
    return np.array([f0, f1]), targets


def _check_training_reduces_error(scale):
    features_array, targets = _make_data(seed=0, scale=scale)

    predictor_inputs, bin_edges, base_prediction = fit_ensemble(features_array, targets, m=100)
    final_pred, _ = predict(features_array, predictor_inputs, bin_edges, base_prediction)

    baseline_rmse = np.sqrt(np.mean((np.mean(targets) - targets) ** 2))
    final_rmse = np.sqrt(np.mean((final_pred - targets) ** 2))

    assert final_rmse < baseline_rmse * 0.5


def test_fit_ensemble_accuracy_ideal_scale():
    # targets already near unit scale -> scaling_factor is close to a no-op
    _check_training_reduces_error(scale=1.0)


def test_fit_ensemble_accuracy_scale_down_needed():
    # large targets (house-price-like) -> scaling_factor divides them down
    _check_training_reduces_error(scale=100_000.0)


def test_fit_ensemble_accuracy_scale_up_needed():
    # tiny targets -> dividing by a tiny std scales them up
    _check_training_reduces_error(scale=0.0001)


def _manual_reference(features_array, targets, m):
    # independent, hand-assembled replica of fit_ensemble's orchestration,
    # built directly from the already-tested lower-level primitives
    scaling_factor = np.std(targets) + 1e-8
    scaled_targets = targets / scaling_factor

    bin_indices = [assign_bins(f)[0] for f in features_array]
    all_feature_bins = dict(enumerate(bin_indices))

    base_prediction = np.mean(scaled_targets)
    y_pred = np.full_like(scaled_targets, base_prediction)
    g, h = compute_gradients_and_hessians(scaled_targets, y_pred)
    splits, leaf_values, node_assignments = fit(g, h, all_feature_bins)

    predictor_inputs = []
    for _ in range(m):
        scaled_leaf_values = leaf_values * learning_rate
        predictor_inputs.append((splits, scaled_leaf_values * scaling_factor))
        y_pred += scaled_leaf_values[node_assignments]
        g, h = compute_gradients_and_hessians(scaled_targets, y_pred)
        splits, leaf_values, node_assignments = fit(g, h, all_feature_bins)

    return predictor_inputs, base_prediction * scaling_factor


def test_fit_ensemble_matches_manual_reference():
    # catches tree-count/off-by-one and splits/leaf_values pairing bugs by
    # diffing against an independently assembled reference, tree by tree
    features_array, targets = _make_data(seed=5, scale=1.0, n=40)
    m = 6

    ref_predictor_inputs, ref_base_prediction = _manual_reference(features_array, targets.copy(), m)
    predictor_inputs, _, base_prediction = fit_ensemble(features_array, targets, m)

    assert len(predictor_inputs) == m
    assert len(predictor_inputs) == len(ref_predictor_inputs)
    assert np.isclose(base_prediction, ref_base_prediction)

    for (splits, leaf_values), (ref_splits, ref_leaf_values) in zip(predictor_inputs, ref_predictor_inputs):
        assert splits == ref_splits
        assert np.allclose(leaf_values, ref_leaf_values, atol=1e-8)


def test_fit_ensemble_does_not_mutate_targets():
    features_array, targets = _make_data(seed=1, scale=1.0)
    targets_before = targets.copy()

    fit_ensemble(features_array, targets, m=5)

    assert np.array_equal(targets, targets_before)
