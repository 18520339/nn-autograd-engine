import numpy as np

def mean_squared_error(y_trues, y_preds):
    return sum((y - y_pred)**2 for y, y_pred in zip(y_trues, y_preds)) / len(y_trues)

def binary_crossentropy(y_trues, y_preds): # Bernoulli cross-entropy
    return -sum(
        y * y_pred.log() + (1 - y) * (1 - y_pred).log()
        for y, y_pred in zip(y_trues, y_preds)
    ) / len(y_trues)

def categorical_crossentropy(y_trues, y_preds):
    return -sum(
        y * y_pred.log() # Call the log method of the tensor
        for y_onehot, y_pred_onehot in zip(y_trues, y_preds)
        for y, y_pred in zip(y_onehot, y_pred_onehot)
    ) / len(y_trues)