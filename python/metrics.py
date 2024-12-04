import numpy as np

def accuracy(y_trues, y_preds):
    if hasattr(y_trues[0], '__len__') and len(y_trues[0]) > 1: # If one-hot encoded
        return sum( # Compare the index of the max value in the one-hot encoded vector
            np.argmax(y_onehot) == max(range(len(y_pred_onehot)), key=lambda i: y_pred_onehot[i].data)
            for y_onehot, y_pred_onehot in zip(y_trues, y_preds)
        ) / len(y_trues)
    return sum(y == y_pred.data for y, y_pred in zip(y_trues, y_preds)) / len(y_trues)