# Tests for gbt.ensemble.predictor: bin_inputs and predict.
import numpy as np

from gbt.ensemble.predictor import predict, bin_inputs

EDGES = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)


def test_bin_inputs_accuracy():
    values = np.array([25.0, 85.0, 5.0, 95.0])

    result = bin_inputs([values], [EDGES])

    assert np.array_equal(result[0], np.array([2, 8, 0, 9]))


def test_bin_inputs_outliers_both_directions():
    values = np.array([-1000.0, 1000.0])

    result = bin_inputs([values], [EDGES])

    assert result[0][0] == -1
    assert result[0][1] == 10


def _synthetic_model():
    bin_edges = [EDGES]
    predictor_inputs = [([(0, 4)], np.array([-5.0, 5.0]))]
    base_prediction = 100.0
    return predictor_inputs, bin_edges, base_prediction


def test_predict_bins_input_correctly():
    predictor_inputs, bin_edges, base_prediction = _synthetic_model()

    feature_arrays = np.array([[25.0, 85.0]])

    final_pred, _ = predict(feature_arrays, predictor_inputs, bin_edges, base_prediction)

    assert np.array_equal(final_pred, np.array([95.0, 105.0]))


def test_predict_outliers_both_directions():
    predictor_inputs, bin_edges, base_prediction = _synthetic_model()

    feature_arrays = np.array([[-1000.0, 1000.0]])

    final_pred, _ = predict(feature_arrays, predictor_inputs, bin_edges, base_prediction)

    assert np.all(np.isfinite(final_pred))
    assert np.array_equal(final_pred, np.array([95.0, 105.0]))
