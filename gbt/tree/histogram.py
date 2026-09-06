# Builds per-node, per-feature gradient/hessian histograms used for split search.

import numpy as np
from numba import njit
from gbt.config import bin_count

@njit(fastmath=True)
def build_feature_histograms(g_array, h_array, node_assignments, feature_bins, num_nodes):

    histogram = np.zeros((num_nodes, bin_count, 2), dtype=np.float64)

    for i in range(len(node_assignments)):
        node = node_assignments[i]
        bin_index = feature_bins[i]
        histogram[node, bin_index, 0] += g_array[i]
        histogram[node, bin_index, 1] += h_array[i]
    return histogram

@njit(fastmath=True)
def update_node_assignments(split_bin, feature_bins, node_assignments):
    new_node_assignments = np.zeros_like(feature_bins, dtype=np.int32)
    for i in range(len(feature_bins)):
        if feature_bins[i] <= split_bin:
            new_node_assignments[i] = 2 * node_assignments[i]
        else:
            new_node_assignments[i] = 2 * node_assignments[i] + 1
    return new_node_assignments

 