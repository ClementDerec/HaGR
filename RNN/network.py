import os
import numpy as np
from itertools import product

from utils import MyUtils

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_derivee = None
        self.epochs = 24
        self.learning_rate = 0.01
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

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
    def fit(self, x_train, y_train,):
        # récupère le nombre de données pour lesquelles il faut faire des prédictions
        samples = len(x_train)
        
        n = round(samples/self.epochs) -1
        
        for i in range(self.epochs) :
            err = 0
            for j in range(i*n,(i+1)*n) :
                # forward propagation
                output = np.array(x_train[j])
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # calcule l'erreur ( uniquement pour l'affichage )
                err += self.loss(y_train[j], output)

                # retroprogation du gradient de l'erreur
                error = self.loss_derivee(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, self.learning_rate)

            # calcule l'erreur moyenne sur tous les échantillons
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, self.epochs, err))
        
        

    def test (self,train,test) :
        samples = len(train)
        result = []
        error = 0
        # fait tourner le réseau pour ttes les données
        for i in range(samples):
            # forward propagation
            output = train[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
            error += self.loss(output,test[i])
            
        print(error/samples)
        
    def hyperparameters_tuning(self, x_train : list[float], y_train : list[float], x_test : list, y_test, hyperparameters : dict[str, list], hyperpamameters_variation : dict[str, list]) -> list:
        """
        Effectue un tuning des hyperparamètres du réseau de neurones en utilisant une approche de recherche par grille.

        Args:
            x_train (list): Données d'entrée pour l'entraînement.
            y_train (list): Données de sortie pour l'entraînement.
            hyperparameters (dict): Dictionnaire contenant les hyperparamètres à ajuster et leurs valeurs possibles.

        Returns:
            list: Liste des résultats de la recherche d'hyperparamètres.
        """

        # Génère toutes les combinaisons possibles d'hyperparamètres
        keys = list(hyperpamameters_variation.keys())
        ranges = [MyUtils.float_range(value[0], value[1] + value[2], value[2]) for value in hyperpamameters_variation.values()]
        combinations = product(*ranges)
        
        self.modify_attr(hyperparameters)
        
        results = []
        for combination in combinations:
            # Modifie les attributs du réseau en fonction des hyperparamètres
            hyper_param_values = dict(zip(keys, combination))
            self.modify_attr(hyper_param_values)

            # Entraîne le réseau avec les hyperparamètres actuels
            self.fit(x_train, y_train)

            # Évalue les performances et stocke les résultats
            predictions = self.predict(x_test)
            loss = self.loss(y_test, predictions)
            dic = self.__dict__
            results.append([[key, dic[key]] for key in hyperpamameters_variation.keys()] + [loss])
            
        name = ""  
        for key in hyperpamameters_variation.keys() :
            name += key + "_" + str(hyperpamameters_variation[key][0]) + "_" + str(hyperpamameters_variation[key][1]) + "_" + str(hyperpamameters_variation[key][2]) + ","
            
        MyUtils.write_json_list(os.path.join(self.base_dir,"logs\\RNN_hyperparameters_tuning" + name + ".json"),results)

        return results
    
    
    def modify_attr(self, modif : dict[str, int]) -> None :
        """modifie les attributs de l'objet en fonction du dictionnaire passé en paramètre

        Args:
            modify (dict[str,int]): dictionnaire contenant les attributs à modifier et leur nouvelle valeur
        """
        for key,value in modif.items() :
            setattr(self,key,value)