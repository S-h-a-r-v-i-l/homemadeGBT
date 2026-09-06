# Builds per-feature histogram bin edges and maps raw feature values to bin indices.

import numpy as np
from numba import njit
from gbt.config import bin_count

def assign_bins(feature_array):
    """
    PYTHON FUNCTION: assign_bins
    Builds histogram bin edges and maps raw feature values to bin indices.

    Args:
        feature_array (np.ndarray): 1D array of feature values.
    """

    percentiles = np.linspace(0, 100, bin_count + 1)
    bin_edges = np.percentile(feature_array, percentiles)

    bin_indices = np.clip(np.digitize(feature_array, bin_edges) - 1, 0, bin_count - 1)
    return bin_indices, bin_edges

@njit
def build_histogram(g_array, h_array, bin_indices):
    """
    PYTHON NJIT FUNCTION: build_histogram
    Builds histogram bin edges and maps raw feature values to bin indices.

    Args:
        g_array (np.ndarray): 1D array of gradients corresponding to the feature values.
        h_array (np.ndarray): 1D array of hessians corresponding to the feature values.
        bin_indices (np.ndarray): 1D array of bin indices for each feature value.
    """
    histogram = np.zeros((bin_count, 2), dtype=np.float64)

    for i in range(len(bin_indices)):
        bin_i = bin_indices[i]
        histogram[bin_i, 0] += g_array[i]
        histogram[bin_i, 1] += h_array[i]

    return histogram



    

