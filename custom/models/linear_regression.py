import math
import numpy as np
from custom.others.data import DataHelper
from custom.math import least_square_residuals

class linear_regression():
    
    def __init__(self, alpha=0.0001, epochs=10000):
        self.alpha = alpha 
        self.epochs = epochs
        self.error_logs = []
        pass
    

    def fit(self, x,y):
        self.x, self.y = DataHelper(x, y)
        try:
            self.w = np.random.random(self.x.shape[1] + 1)
        except:
             self.w = np.random.random(2)
        self.n = self.x.shape[0]
        self.x = np.apply_along_axis(self._append_bias, 1, self.x) 
        self.w = least_square_residuals.LSR(self.x, self.y)
        return self
        
        
    def _append_bias(self, xi):
        return np.append(xi, [1])


    def predict(self, x_test):
        self.x_test=x_test
        self.x_test = np.apply_along_axis(self._append_bias, 1, self.x_test)
        self.predicted_test = self.x_test.dot(self.w)
        return self.predicted_test
    
    