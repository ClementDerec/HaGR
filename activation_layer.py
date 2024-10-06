from layer import Layer

# hérite de la classe layer
class ActivationLayer(Layer):
    def __init__(self, activation, activation_derivee):
        self.activation = activation
        self.activation_prime = activation_derivee

    # retourne la sortie apres la fonction d'activation
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # retourne input_error=dE/dX pour une erreur donnée output_error=dE/dY.
    # learning_rate n'est pas utilisé pck il n'y a pas de parametres apprenants.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error
