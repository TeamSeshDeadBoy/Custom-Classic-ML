import numpy as np

def Accuracy(true, pred):
    return (1 / true.shape[0]) * np.sum(np.equal(true, pred))

def ErrorRate(true, pred):
    return 1 - Accuracy(true, pred)

def _prepare_data(true, pred):
    assert np.unique(pred).shape[0] <= 2, "The classification should be binary"
    assert np.unique(true).shape[0] <= 2, "The classification should be binary"
    classes = np.unique(true)
    true = np.where(true == classes[0], 1, 0)
    pred = np.where(pred == classes[0], 1, 0)
    return true, pred

def _TP(true, pred, prepared=False):
    if not prepared:
        true, pred = _prepare_data(true, pred)
    return np.sum(np.logical_and(true == 1, pred == 1))
        
def _TN(true, pred, prepared=False):
    if not prepared:
        true, pred = _prepare_data(true, pred)
    return np.sum(np.logical_and(true == 0, pred == 0))

def _FN(true, pred, prepared=False):
    if not prepared:
        true, pred = _prepare_data(true, pred)
    return np.sum(np.logical_and(true == 1, pred == 0))

def _FP(true, pred, prepared=False):
    if not prepared:
        true, pred = _prepare_data(true, pred)
    return np.sum(np.logical_and(true == 0, pred == 1))

def Precision(true, pred):
    true, pred = _prepare_data(true, pred)
    return _TP(true, pred, True) / (_TP(true, pred, True) + _FP(true, pred, True))

def Recall(true, pred):
    true, pred = _prepare_data(true, pred)
    return _TP(true, pred, True) / (_TP(true, pred, True) + _FN(true, pred, True))

def ConfusionMatrix(true, pred):
    true, pred = _prepare_data(true, pred)
    return [[_TP(true, pred, True), _FN(true, pred, True)], [_FP(true, pred, True), _TN(true, pred, True)]]
    