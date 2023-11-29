import numpy as np
from custom.metrics import regression_metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def test(target, predictions):
    assert np.allclose(regression_metrics.MSE(target, predictions), mean_squared_error(target, predictions), rtol=1e-3), "MSE defers from sklearn"
    assert np.allclose(regression_metrics.RMSE(target, predictions), mean_squared_error(target, predictions, squared=False), rtol=1e-3), "RMSE defers from sklearn"
    assert np.allclose(regression_metrics.MAE(target, predictions), mean_absolute_error(target, predictions), rtol=1e-3), "MAE defers from sklearn"
    assert np.allclose(regression_metrics.MAPE(target, predictions), mean_absolute_percentage_error(target, predictions), rtol=1e-3), "MAPE defers from sklearn"
    
    print("Regression metrics tests passed successfully. Metrics are the same as sklearn.")
