import numpy as np
from custom.metrics import classification_metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

def test(target, predictions):
    assert np.allclose(classification_metrics.Accuracy(target, predictions), accuracy_score(target, predictions), rtol=1e-3), "Accuracy defers from sklearn"
    assert np.allclose(classification_metrics.ErrorRate(target, predictions), (1 - accuracy_score(target, predictions)), rtol=1e-3), "Error Rate defers from sklearn"
    assert np.allclose(classification_metrics.Precision(target, predictions),precision_score(target, predictions, pos_label="blue"), rtol=1e-3), "Precision defers from sklearn"
    assert np.allclose(classification_metrics.Recall(target, predictions),recall_score(target, predictions, pos_label="blue"), rtol=1e-3), "Recall defers from sklearn"
    assert np.array_equal(classification_metrics.ConfusionMatrix(target, predictions), confusion_matrix(target,predictions)), "Confusion Matrix defers from sklearn"
    print("Classification metrics tests passed successfully. Metrics are the same as sklearn. (5/5) (Accuacy, Error Rate, Precision, Recall, Confusion Matrix)")
