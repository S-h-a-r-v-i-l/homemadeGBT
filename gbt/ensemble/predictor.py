# Applies a fitted binner and sequence of trees to produce predictions on new data.
import numpy as np
from gbt.data.binning import assign_bins
from gbt.tree.histogram import update_node_assignments

def predict(feature_arrays, tree_ensemble, bin_edges, base_prediction):
    feature_bins = bin_inputs(feature_arrays, bin_edges)
    prediction = base_prediction
    all_predictions = []
    for tree in tree_ensemble:
        splits, leaf_values = tree
        node_assignments = np.zeros(len(feature_arrays[0]), dtype=np.int32)
        for feature, split_bin in splits:
            node_assignments = update_node_assignments(split_bin, feature_bins[feature], node_assignments)
        prediction += leaf_values[node_assignments]
        all_predictions.append(prediction.copy())

    return prediction, all_predictions

def bin_inputs(feature_arrays, bin_edges):
    binned_inputs = []
    for feature, edges in zip(feature_arrays, bin_edges):
        binned_feature = np.digitize(feature, edges) - 1
        binned_inputs.append(binned_feature)
    return binned_inputs