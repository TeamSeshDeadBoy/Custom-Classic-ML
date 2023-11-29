import numpy as np
from custom.metrics import regression_metrics

target = np.array([34, 37, 44, 47, 48, 48, 46, 43, 32, 27, 26, 24])
predictions = np.array([37, 40, 46, 44, 46, 50, 45, 44, 34, 30, 22, 23])

print('MSE: ', regression_metrics.MSE(target, predictions))
print('RMSE: ',regression_metrics.RMSE(target, predictions))
print('MAE: ', regression_metrics.MAE(target, predictions))
print('MAPE: ', regression_metrics.MAPE(target, predictions))
print('SMAPE: ', regression_metrics.SMAPE(target, predictions))