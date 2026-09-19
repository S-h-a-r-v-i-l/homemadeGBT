# Symmetric (oblivious) tree: same split rule applied to every node at a given depth.
from gbt.tree.histogram import update_node_assignments
from gbt.tree.splitter import find_level_split
from gbt.config import max_depth
from gbt.config import lambda_
import numpy as np

def fit(g_array, h_array, all_feature_bins):
    node_assignments = np.zeros(g_array.shape[0], dtype=np.int32)
    splits = []
    for depth in range(max_depth):
        feature, split_bin, gain = find_level_split(g_array, h_array, node_assignments, all_feature_bins, 2**depth)
        if(feature is None or split_bin == -1):
                    break
        splits.append((feature, split_bin))
        node_assignments = update_node_assignments(split_bin, all_feature_bins[feature], node_assignments)

    num_leaves = 2 ** len(splits)
    G, H = np.zeros(num_leaves), np.zeros(num_leaves)
    for i in range(len(node_assignments)):
        leaf = node_assignments[i]
        G[leaf] += g_array[i]
        H[leaf] += h_array[i]
    leaf_values = -G / (H + lambda_)

    return splits, leaf_values, node_assignments


    

    

        

    