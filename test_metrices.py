import numpy as np
from custom.tests import classification_metrics_test, regression_metrics_test

target_cl =      np.array(["red", "red", "red", "red", "blue", "red", "blue", "blue", "blue", "blue", "blue", "blue"])
predictions_cl = np.array(["red","red","red","red","red", "blue", "blue", "blue", "blue", "blue", "blue", "blue"])
classification_metrics_test.test(target_cl, predictions_cl)

target = np.array([1, 2, 3, 4, 5, 6, 7, 8])
predictions = np.array([2, 3, 3, 4, 4, 5, 7, 7])
regression_metrics_test.test(target, predictions)