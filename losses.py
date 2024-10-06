import numpy as np

# fonction de perte et sa derivée
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_derivee(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size
