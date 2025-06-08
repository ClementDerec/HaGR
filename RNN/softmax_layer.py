import numpy as np


class SoftmaxLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.inputs = None
        self.output = None

    def forward_propagation(self, inputs):
        self.inputs = inputs
        z = np.dot(inputs, self.weights) + self.biases
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        self.output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.output

    def backward_propagation(self, grad_output, learning_rate):
        grad_z = self.output - grad_output
        grad_weights = np.dot(self.inputs.T, grad_z) / self.inputs.shape[0]
        grad_biases = np.sum(grad_z, axis=0, keepdims=True) / self.inputs.shape[0]
        grad_inputs = np.dot(grad_z, self.weights.T)
        
        # Update weights and biases using the learning rate
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases
        
        return grad_inputs