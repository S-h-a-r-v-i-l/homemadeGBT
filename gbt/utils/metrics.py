# Evaluation metrics (RMSE, MAE, logcosh) for tracking training/validation performance.

from gbt.loss.logcosh import logcosh_loss
import numpy as np

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def logcosh(y_true, y_pred):
    return logcosh_loss(y_true, y_pred)
