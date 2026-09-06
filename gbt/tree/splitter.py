import numpy as np
from numba import njit

from gbt.config import max_depth
from gbt.config import gamma
from gbt.config import lambda_
from gbt.tree.histogram import build_feature_histograms


@njit(fastmath=True)
def find_best_feature_split(featuregrams):
    num_nodes = featuregrams.shape[0]
    bin_count = featuregrams.shape[1]

    depth = 0
    remaining = num_nodes
    while remaining > 1:
        remaining >>= 1
        depth += 1

    if depth >= max_depth:
        return -1, 0.0

    total_grad = np.zeros(num_nodes)
    total_hess = np.zeros(num_nodes)
    
    for n in range(num_nodes):
        for b in range(bin_count):
            total_grad[n] += featuregrams[n, b, 0]
            total_hess[n] += featuregrams[n, b, 1]

    cum_grad = np.zeros(num_nodes)
    cum_hess = np.zeros(num_nodes)

    parent_score = np.zeros(num_nodes)
    for n in range(num_nodes):
        g_p = total_grad[n]
        h_p = total_hess[n]
        parent_score[n] = 0.5 * (g_p * g_p) / (h_p + lambda_)

    best_gain = 0.0
    best_bin = -1

    for b in range(bin_count - 1):
        curr_gain = 0.0
        for n in range(num_nodes):
            cum_grad[n] += featuregrams[n, b, 0]
            cum_hess[n] += featuregrams[n, b, 1]

            g_l = cum_grad[n]
            h_l = cum_hess[n]
            g_r = total_grad[n] - g_l
            h_r = total_hess[n] - h_l

            gain = 0.5 * ((g_l * g_l) / (h_l + lambda_) + (g_r * g_r) / (h_r + lambda_)) - parent_score[n]
            curr_gain += gain

        curr_gain -= gamma

        if curr_gain > best_gain:
            best_gain = curr_gain
            best_bin = b

    return best_bin, best_gain


def find_level_split(g_array, h_array, node_assignments, all_feature_bins, num_nodes):
    best_bin, best_gain, best_feature = -1, 0.0, None
    for feature, feature_bins in all_feature_bins.items():
        featuregrams = build_feature_histograms(g_array, h_array, node_assignments, feature_bins, num_nodes)
        curr_bin, curr_gain = find_best_feature_split(featuregrams)
        if curr_gain > best_gain:
            best_gain = curr_gain
            best_bin = curr_bin
            best_feature = feature

    if best_gain <= 0.0:
        return None, -1, 0.0
    return best_feature, best_bin, best_gain

        