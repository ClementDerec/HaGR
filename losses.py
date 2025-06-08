import numpy as np
from utils import MyUtils

# fonction de perte et sa derivée
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_derivee(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

def cross_entropy(true_label, predicted_probabilities_for_each_class):
    """Cross-entropy loss function.

    Args:
        true_label (np.array): One-hot encoded vector.
        predicted_probabilities_for_each_class (np.array): Probabilities generated with a softmax function.

    Returns:
        float: Cross-entropy loss value.
    """
    epsilon = 1e-15  # To avoid log(0)
    predicted_probabilities_for_each_class = np.clip(predicted_probabilities_for_each_class, epsilon, 1 - epsilon)
    return -np.sum(true_label * np.log(predicted_probabilities_for_each_class)) / true_label.shape[0]
    
    
    
def cross_entropy_derivee (predicted_values,true_probabilities):
    """_summary_

    Args:
        predicted_values (np.array): predicted values for 1 vector
        real_probabilities (np.array): one_hot encoded vector for 1 vector
        
    """
    
    return predicted_values-true_probabilities


def huber_loss(y_true, y_pred, delta=1.0):
    """Huber loss function.
    
    Args:
        y_true (np.array): True values.
        y_pred (np.array): Predicted values.
        delta (float): Threshold for the loss function.
        
    Returns:
        np.array: Huber loss value.
    """
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * np.square(error)
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.where(is_small_error, squared_loss, linear_loss)

def huber_loss_derivative(y_true, y_pred, delta=1.0):
    """Derivative of the Huber loss function.
    
    Args:
        y_true (np.array): True values.
        y_pred (np.array): Predicted values.
        delta (float): Threshold for the loss function.
        
    Returns:
        np.array: Derivative of the Huber loss value.
    """
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    return np.where(is_small_error, error, delta * np.sign(error))