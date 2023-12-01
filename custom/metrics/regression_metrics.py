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

def RSQ(true, pred):
    mean = np.mean(true)
    ss_mean = np.sum(np.square(true - mean))
    ss_fit =  np.sum(np.square(true - pred))
    return (ss_mean - ss_fit) / ss_mean