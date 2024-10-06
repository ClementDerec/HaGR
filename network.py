import numpy as np
class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_derivee = None

    # ajoute couche au réseau
    def add(self, layer):
        self.layers.append(layer)

    # definit la fonction de perte utilisee 
    def use(self, loss, loss_derivee):
        self.loss = loss
        self.loss_derivee = loss_derivee

    # predit une sortie por l'entree "input_data"
    def predict(self, input_data):
        # récupère le nombre de données pour lesquelles il faut faire des prédictions
        samples = len(input_data)
        result = []

        # fait tourner le réseau pour ttes les données
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # entraine le réseau
    def fit(self, x_train, y_train, epochs, learning_rate):
        # récupère le nombre de données pour lesquelles il faut faire des prédictions
        samples = len(x_train)

        # boucle pour entrainer le réseau sur ttes les données 'epochs' fois
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = np.array(x_train[j])
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # calcule l'erreur ( uniquement pour l'affichage )
                err += self.loss(y_train[j], output)

                # retroprogation du gradient de l'erreur
                error = self.loss_derivee(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calcule l'erreur moyenne sur tous les échantillons
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))
