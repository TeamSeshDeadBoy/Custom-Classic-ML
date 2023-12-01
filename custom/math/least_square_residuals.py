import numpy as np

def LSR(x, y):
    x_t = np.transpose(x)
    return np.linalg.inv(np.dot(x_t, x)).dot(np.dot(x_t, y))