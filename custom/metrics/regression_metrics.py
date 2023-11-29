import numpy as np

def MSE(true, pred):
    return (1 / true.shape[0]) * np.sum(np.square(true - pred))

def RMSE(true, pred):
    return np.sqrt(MSE(true, pred))

def MAE(true, pred):
    return  (1 / true.shape[0]) * np.sum(np.abs(true - pred))

def MAPE(true, pred):
    return (1 / true.shape[0]) * np.sum(np.abs(true - pred) / true)

def SMAPE(true, pred):
    return (1 / true.shape[0]) * np.sum((2 * np.abs(true - pred)) / (true + pred))

def RSQUARED(true, pred):
    n = len(true)
    return np.square((n * np.sum(true * pred) - np.sum(true) * np.sum(pred)) / np.sqrt((n * np.sum(np.square(true)) - np.square(np.sum(true))) * (n * np.sum(np.square(pred)) - np.square(np.sum(pred)))))