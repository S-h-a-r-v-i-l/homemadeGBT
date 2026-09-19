#logcosh, a continuous huber alternative loss function, is a smooth approximation to the absolute error loss function. It is defined as log(cosh(x)), where x is the difference between the predicted and actual values. The logcosh loss function is less sensitive to outliers than the mean squared error loss function, making it a good choice for regression problems with noisy data.

from numba import njit
import numpy as np

@njit
def logcosh_loss(y_true, y_pred):
    abs_x = np.abs(y_pred - y_true)
    return np.mean(abs_x - np.log(2.0) + np.log1p(np.exp(-2.0 * abs_x)))

@njit
def compute_gradients_and_hessians(y_true, y_pred):
    gradient = np.tanh(y_pred - y_true)
    hessians = 1 - gradient ** 2
    return gradient, hessians