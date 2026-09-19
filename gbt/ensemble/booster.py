# Main training loop: bins data, then iteratively fits oblivious trees to loss gradients/hessians.
from gbt.data.binning import assign_bins
import numpy as np
from gbt.config import learning_rate

from gbt.loss.logcosh import compute_gradients_and_hessians
from gbt.tree.oblivious_tree import fit

def _fit_ensemble(features_array, targets, m):
    bin_indices = []
    bin_edges = []
    for feature in features_array:
        if len(feature) != len(targets):
            raise ValueError("All features must have the same length as targets.")
        curr_bin_indices, curr_bin_edges = assign_bins(feature)
        bin_indices.append(curr_bin_indices)
        bin_edges.append(curr_bin_edges)

    all_feature_bins = dict(enumerate(bin_indices))

    base_prediction = np.mean(targets)
    y_pred = np.full_like(targets, base_prediction)
    base_g, base_h = compute_gradients_and_hessians(targets, y_pred)
    splits, leaf_values, node_assignments = fit(base_g, base_h, all_feature_bins)

    predictor_inputs = []

    for i in range(m):
        scaled_leaf_values = leaf_values * learning_rate
        predictor_inputs.append((splits, scaled_leaf_values))
        y_pred += scaled_leaf_values[node_assignments]
        g, h = compute_gradients_and_hessians(targets, y_pred)
        splits, leaf_values, node_assignments = fit(g, h, all_feature_bins)
        

    return predictor_inputs, bin_edges, base_prediction


def fit_ensemble(features_array, targets, m):
    scaling_factor = np.std(targets) + 1e-8
    scaled_targets = targets / scaling_factor
    predictor_inputs, bin_edges, base_prediction = _fit_ensemble(features_array, scaled_targets, m)
    base_prediction *= scaling_factor
    predictor_inputs = [(splits, leaf_values * scaling_factor) for splits, leaf_values in predictor_inputs]
    return predictor_inputs, bin_edges, base_prediction