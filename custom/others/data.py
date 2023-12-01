import numpy as np

def DataHelper(x, y):
    print("-" * 23)
    print("Inputed data info:")
    if type(x) != np.ndarray:
        x = np.array(x)
    if type(y) != np.ndarray:
        y = np.array(y)
    print("Y. shape: ", y.shape, "type: ", type(y), "Ex:", y[0])
    print("X. shape: ", x.shape, "type: ", type(x), "Ex:", x[0])
    print("-" * 23)
    return x, y