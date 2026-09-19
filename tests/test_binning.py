# Tests for gbt.data.binning.assign_bins.
import numpy as np

from gbt.data.binning import assign_bins
from gbt.config import bin_count


def test_assign_bins_accuracy():
    feature_array = np.arange(100, dtype=float)

    bin_indices, bin_edges = assign_bins(feature_array)

    assert len(bin_edges) == bin_count + 1
    assert bin_indices.min() >= 0
    assert bin_indices.max() <= bin_count - 1
    assert bin_indices[0] == 0                  
    assert bin_indices[-1] == bin_count - 1     

    # evenly spaced data should split into evenly sized bins
    counts = np.bincount(bin_indices, minlength=bin_count)
    assert (counts == 10).all()


def test_assign_bins_outlier_high():
    normal_values = np.random.RandomState(0).uniform(0, 100, size=50)
    feature_array = np.append(normal_values, 1_000_000.0)

    bin_indices, _ = assign_bins(feature_array)

    assert bin_indices[-1] == bin_count - 1


def test_assign_bins_outlier_low():
    normal_values = np.random.RandomState(0).uniform(0, 100, size=50)
    feature_array = np.append(normal_values, -1_000_000.0)

    bin_indices, _ = assign_bins(feature_array)

    assert bin_indices[-1] == 0


def test_assign_bins_constant_values_does_not_crash():
    feature_array = np.full(20, 42.0)

    bin_indices, bin_edges = assign_bins(feature_array)

    assert len(bin_indices) == len(feature_array)
    assert len(bin_edges) == bin_count + 1
