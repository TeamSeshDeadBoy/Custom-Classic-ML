import numpy as np
from custom.metrics import classification_metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

target =      np.array(["red", "red", "red", "red", "blue", "red", "blue", "blue", "blue", "blue", "blue", "blue"])
predictions = np.array(["red","red","red","red","red", "blue", "blue", "blue", "blue", "blue", "blue", "blue"])

print('Accuracy: ', classification_metrics.Accuracy(target, predictions), 'sklearn: ', accuracy_score(target, predictions))
print('Error Rate: ',classification_metrics.ErrorRate(target, predictions), 'sklearn: ', (1 - accuracy_score(target, predictions)))
print('Precision: ', classification_metrics.Precision(target, predictions), 'sklearn: ', precision_score(target, predictions, pos_label="blue"))
print('Recall: ', classification_metrics.Recall(target, predictions), 'sklearn: ', recall_score(target, predictions, pos_label="blue"))
print('Confusion Matrix: ', classification_metrics.ConfusionMatrix(target, predictions), 'Same with sklearn: ', np.array_equal(classification_metrics.ConfusionMatrix(target, predictions), confusion_matrix(target,predictions)))
