import numpy as np
from utils import MyUtils

# fonction de perte et sa derivée
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_derivee(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

def cross_entropy (number_classes,number_sample_batch,true_label,predicted_probabilities_for_each_class) :
    """_summary_

    Args:
        number_classes (int): number of classes
        number_sample_batch (int): number of sample in the batch
        true_label (np.array): one_hot encoded vector
        predicted_probabilities_for_each_class (np.array): it has to be a probability generated with a softmax function
    """
    sum = 0
    
    for i in range(number_sample_batch -1) :
        for c in range(number_classes -1) :
            sum += true_label[i][c]*np.log10(predicted_probabilities_for_each_class[i][c])
    return sum
    
    
    
def cross_entropy_derivee (predicted_values,true_probabilities):
    """_summary_

    Args:
        predicted_values (np.array): predicted values for 1 vector
        real_probabilities (np.array): one_hot encoded vector for 1 vector
        
    """
    
    return predicted_values-true_probabilities