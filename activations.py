import numpy as np

# fonction d'activation et sa dérivée
def tanh(x):
    return np.tanh(x)

def tanh_derivee(x):
    return 1-np.tanh(x)**2
