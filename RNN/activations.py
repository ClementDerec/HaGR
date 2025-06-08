import numpy as np

# fonction d'activation et sa dérivée, agisse sur un vecteur
def tanh(x):
    return np.tanh(x)

def tanh_derivee(x):
    return 1-np.tanh(x)**2

def relu(x) :
    output = []
    for y in x[0] :
        if y<0 : 
            output.append(0)
        else : 
            output.append(y)
            
    return np.array([output]) 
   
def relu_derivee(x):
    output = []
    for y in x[0] :
        if y<=0 : 
            output.append(0)
        else : 
            output.append(1)
    return np.array([output])

def sigmoid(x : list[list[float]]) -> list[list[float]]:
    """
    Fonction d'activation sigmoïde appliquée à chaque élément de la matrice d'entrée.
    Args:
        x (list[list[float]]): Matrice d'entrée.
    Returns:
        list[list[float]]: Matrice de sortie après application de la fonction sigmoïde.
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivee(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    """"
    Applies the softmax function to the input array.
    Args:
        x (numpy.ndarray): Input array (vector or matrix).
        
    Returns:
        numpy.ndarray: Softmax-transformed array with the same shape as input.
    """
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def softmax_derivee(x):
    """
    Computes the derivative of the softmax function for a multi-dimensional array.
    Args:
        x (numpy.ndarray): Input array (matrix).
        
    Returns:
        numpy.ndarray: Derivative of the softmax function with the same shape as input.
    """
    s = softmax(x)
    jacobian = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[1]):
                if j == k:
                    jacobian[i, j, k] = s[i, j] * (1 - s[i, k])
                else:
                    jacobian[i, j, k] = -s[i, j] * s[i, k]
    return jacobian