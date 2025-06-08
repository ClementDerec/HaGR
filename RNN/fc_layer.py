from RNN.layer import Layer
import numpy as np

# herite de la classe layer 
class FCLayer(Layer):
    # input_size = nombre de neruones d'entree
    # output_size = nombre de neurones de sortie
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # donne la sortie pour l'entree "input_data"
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    
    # calcule dE/dW, dE/dB pour une erreur donnée output_error=dE/dY. Retourne input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error car pas de fonction d'activation

        # mets à jour les poids
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
    
    