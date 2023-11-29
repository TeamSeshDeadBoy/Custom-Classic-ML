import math
import numpy as np

class linear_regression():
    
    def __init__(self, alpha=0.0001, epochs=10000):
        self.alpha = alpha 
        self.epochs = epochs
        self.error_logs = []
        pass
    

    def fit(self, x,y):
        self.x = x
        self.y = y
        
        self.w = np.random.random(self.x.shape[1])
        self.b = 0.00001
        
        self.n = self.x.shape[0]
        self._lin_reg_grad()
        


    def _lin_reg_grad(self):
        epoch = 0                                    # initialize epoch 0
        while epoch < self.epochs:                   # self.epochs is a hyperparamether we chose (number of iterations)
            f = self.x.dot(self.w) + self.b          # f is a predicted y vector (y = x * w + b)
            err = f - self.y                         # error is an error between predicted y and real y
            grad = (2 * self.x.T @ err ) / self.n    # calculate gradient descent of MSE loss fucntion for weights
            grad_b = np.sum(err) / self.n            # calculate gradient descent of MSE loss fucntion for bias
            self.w -= self.alpha * grad              # self.alpha is a hyperparamether (a.k.a learning rate) we multiply by our antigradient (gradient * -1) to have a value that shows us where we need to move our weights (increase \ decrease) for them to be closer to real weights that we need
            self.b -= self.alpha * grad_b            # same goes for bias
            
            print('-' * 23)                          # print number of epoch, weights, bieas
            print("Epoch: ", epoch)
            print('W: ', self.w)
            print("B: ", self.b)
            
            mse = self.metric(f)                    # calculate MSE metric for our predicted data
            self.mse.append(mse)                    # store this loss function value in our array
            print("MSE: ", mse)                     # print it out
            epoch += 1                              # move to next epoch (repeat the same again)

        
    
    # metric is basically a simple MSE function that calculates A Mean Square Error for our predicted data
    def metric(self, pred):
        return (1 / self.n) * np.sum(np.square(self.y - pred))
        
    
    # Predict function
    # take our x and multiply it by calculated weights (calculate y = w * x(test) + b) where we have w and b already calculated in gradient descent
    def predict(self, x_test, y_test):
        self.x_test=x_test
        self.y_test=y_test
        
        self.predicted_test = self.x_test.dot(self.w) + self.b    # calculate predicted y
        err_vec = self.predicted_test - self.y_test               # take an error (difference between real y_test and y that is predicted by our model)
        return (1 / self.x_test.shape[0]) * np.sum(np.square(err_vec)) # print out MSE for our errors
    
    