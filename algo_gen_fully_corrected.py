import random

from Tree import Node
from operator import itemgetter
from utils import MyUtils
import os
from itertools import product
from typing import Optional

class net_gen :
    
    
    def __init__(self, mutation_rate : int, crossover_rate : int,pop_size : int, depth : int, data : list[str], K : int, parents_rate : int, number_iteration : int, elitism_rate: int, mutation_node_rate : int) :
        """initialise les paramètres de l'algorithme génétique

        Args:
            mutation_rate (int): taux de mutation chez les parents
            crossover_rate (int): taux de croisement chez les parents
            pop_size (int): taille de la population
            depth (int): profondeur maximale de l'arbre
            data (list[str]): liste des coordonnées des points repères sous la forme 'x1','x2','y1','y2'
            K (int): hyperparamètre du tournoi (nombre de participants du tournoi, choisi le meilleur parmis K)
            parents_rate (int): taux de parents (elements subissant les muatations) dans le population suivante, le reste est aléatoire 
            number_iteration (int): nombre d'itérations de l'algorithme génétique avant de retourner le meilleur arbre
            elitism_rate (int): pourcentage de la population qui est conservé pour la génération suivante
            mutation_node_rate (int): taux de mutation chez les noeuds
        """
        self.mutation_rate = mutation_rate 
        self.crossover_rate = crossover_rate
        self.pop_size = pop_size
        self.depth = depth
        self.comparators = ['<','>']
        self.op_logic = ['OR','AND']
        self.data = data
        self.K = K
        self.parents_rate = parents_rate
        self.number_iteration = number_iteration
        self.elitism_rate = elitism_rate
        self.mutation_node_rate = mutation_node_rate
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
    def mutate_logic(self, tree_node : Node) :  
        """fait muter un arbre en changeant les opérateurs logiques. Ne retourne pas l'arbre mais le modifie. Si l'arbre est vide la fonction ne fait rien

        Args:
            tree_node (Node): arbre à muter
        """
        if tree_node != None and tree_node.value is not None:
            if random.randint(1,100) <= self.mutation_node_rate :
                if tree_node.value in self.comparators :
                    if tree_node.value == '>' :
                        tree_node.value = '<'
                    else :
                        tree_node.value = '>'

                elif tree_node.value in self.op_logic :
                    if tree_node.value == 'AND' :
                        tree_node.value = 'OR'
                    else :
                        tree_node.value = 'AND'

            self.mutate_logic(tree_node.left)
            self.mutate_logic(tree_node.right)

        
    def mutate_values(self, tree_node : Node) :
        """fait muter un arbre en changeant les valeurs des noeuds. Ne retourne pas l'arbre mais le modifie. Si l'arbre est vide la fonction ne fait rien

        Args:
            tree_node (Node): arbre à muter
        """
        if tree_node != None and tree_node.value is not None:
            if random.randint(1,100) <= self.mutation_node_rate :
                 
                if tree_node.value in self.data :
                    i = random.randint(0,len(self.data)-1)
                    tree_node.value = self.data[i]
                    
            self.mutate_values(tree_node.left)
            self.mutate_values(tree_node.right)
    
    def count_node(self, tree_node : Node) :
        """Parses the tree to find a specific node based on a limit.

        Args:
            tree_node (Node): The root node of the tree to parse.
            limit (int): The index of the node to find.

        Returns:
            tuple[int, Node]: A tuple where the first element is the count of nodes traversed,
                              and the second is the selected node or None if not found.
                              and the second element is the node found at the specified index 
                              (or None if not found).
        """
        if tree_node != None :
            if tree_node.value in self.op_logic :
                val = 1 + self.count_node(tree_node.left) + self.count_node(tree_node.right)
                return val
            else :
                return 0    
        else :
            return 0
        
    def get_selected_node(self, tree_node: "Node", index: int) -> Optional["Node"]:
        """
        Retourne le nœud logique situé à la position `index` dans un parcours en profondeur (ordre préfixe).
        
        Args:
            tree_node (Node): Racine de l'arbre.
            index (int): Index du nœud logique à trouver.

        Returns:
            Node | None: Le nœud logique à l'index donné, ou None si non trouvé.
        """
        self._current_index = 0
        return self._get_node_at_index(tree_node, index)

    def _get_node_at_index(self, node: Optional["Node"], target_index: int) -> Optional["Node"]:
        if node is None:
            return None

        if node.value in self.op_logic:
            if self._current_index == target_index:
                return node
            self._current_index += 1

        # Explorer gauche
        found = self._get_node_at_index(node.left, target_index)
        if found:
            return found

        # Explorer droite
        return self._get_node_at_index(node.right, target_index)  
                    
    def replaceInTree(self, tree_node : Node, replacedNode : Node, index : int) :
        if tree_node != None :
            if tree_node.value in self.op_logic :
                (nbr_g,gauche) = self.replaceInTree(tree_node.left,replacedNode,index)
                (nbr_d,droite) = self.replaceInTree(tree_node.right,replacedNode,index)
                val = nbr_d + nbr_g + 1
                if val == index :
                    return (val,replacedNode)
                else :
                    arbre = Node(tree_node.value)
                    arbre.left= gauche
                    arbre.right = droite 
                    return (val,arbre)
            else :
                return(0,tree_node)
        else :
            return(0,None)
        
    def replace(self, tree_node1 : Node, tree_node2 : Node):
        """remplace les valeurs et les fils des deux arbres. Si un des arbres est vide la fonction ne fait rien.

        Args:
            tree_node1 (Node): le premier arbre
            tree_node2 (Node): le deuxième arbre
        """
        if tree_node1 is None or tree_node2 is None:
            # If either tree_node1 or tree_node2 is None, do nothing
            return
        
        # on remplace le noeud de l'arbre 1 par le noeud de l'arbre 2 et vice versa
        # on fait une copie des deux noeuds pour ne pas perdre les valeurs 
        ar1 = Node.copy(tree_node1)
        ar2 = Node.copy(tree_node2)
        tree_node1.value = ar2.value
        tree_node1.left = ar2.left
        tree_node1.right = ar2.right
        tree_node2.value = ar1.value
        tree_node2.left = ar1.left
        tree_node2.right = ar1.right
            
    def crossover(self, root_node1 : Node, root_node2 : Node) :
        """
        
        Args:
            root_node1 (Node): The root node of the first tree.
            root_node2 (Node): The root node of the second tree.

        Behavior:
            - Randomly selects a node from each tree.
            - Swaps the selected nodes and their subtrees between the two trees.
            - Ensures the indices for node selection are within valid ranges.
        """
        count1, count2 = self.count_node(root_node1), self.count_node(root_node2)
        max_nodes = count1 + count2
        # Randomly select two indices for the nodes to be swapped
        i = random.randint(0, max(0, count1 - 1))
        j = random.randint(0, max(0, count2 - 1))

        
        
        # Ensure indices are within the range of actual nodes
        i = min(i, count1)
        j = min(j, count2)
        
        ar1, ar2 = self.get_selected_node(root_node1, i), self.get_selected_node(root_node2, j)
        
        if ar1 is not None and ar2 is not None:
            self.replace(ar1, ar2)
    
    
    def generate_random_tree (self) -> list[Node] :
        l = []
        for i in range(self.pop_size) :
            tree = Node.generate_root_node()
            tree.left = Node.generate_tree(self.depth,self.data)
            tree.right = Node.generate_tree(self.depth,self.data)
            l.append(tree)
        return l
    
    def return_fitness(self, tree: Node, desired_class: int, landmarks_list: list[tuple[dict[str, int], int]]) -> float:
        """
        Calcule la fitness d’un arbre décisionnel basé sur une liste de points repères.

        La fitness est le nombre de prédictions correctes divisées par le nombre total d'exemples.
        Si l’arbre prédit correctement la classe (booléen), on compte +1.
    
        Args:
            tree (Node): L’arbre à évaluer.
            desired_class (int): La classe attendue (cible à détecter).
            landmarks_list (list[tuple[dict[str, int], int]]): Liste de tuples (features, label)
                - features : dictionnaire des coordonnées (e.g. {"x1": 1, "x2": 2, "y1": 3, "y2": 4})
                - label : classe réelle de cet exemple (entier)

        Returns:
            float: Score de fitness (entre 0.0 et 1.0)
        """
        if not landmarks_list:
            return 0.0

        correct = 0
        for features, true_class in landmarks_list:
            prediction = tree.evaluate(features)  # retourne True/False
            expected = (true_class == desired_class)
            if prediction == expected:
                correct += 1

        return correct / len(landmarks_list)


    
    def tournament_selection (self, list_of_tree : list[tuple[Node, float]], landmarks_list : list[tuple[dict[str,int], float]]) -> list[ (Node, float) ]:
        """Select the best trees from a list of tuples containing trees and their fitness values.

        Args:
            list_of_tree (list[tuple[Node, float]]): List of tuples where each tuple contains a tree and its fitness value.
            landmarks_list (list[tuple[dict[str,int],float]]): List of landmarks with their associated fitness values.
            
        Returns:
            list[(Node,float)]: The list of selected trees for the next generation.
        """
        selected_tree = []
        
        for i in range(round(self.parents_rate * len(list_of_tree) / 100)):  # select parents_rate trees
            # Select K random tuples (tree, fitness) directly using random.sample
            list_fitness = random.sample(list_of_tree, self.K)
            
            # Sort the sampled tuples by fitness in descending order
            list_fitness.sort(reverse=True, key=itemgetter(1))
            
            # Append the tree with the highest fitness to the selected_tree list
            selected_tree.append(list_fitness[0])
        
        return selected_tree
    
    
    
    def fitness(self, tree: Node, points_reperes):
        return tree.evaluate(points_reperes)
    
    def elitism (self, list_of_tree : list[Node,float], landmarks_list : list[ tuple[dict[str,int], float]]) -> list[ (Node,float)] :
        """Select the best trees from a list of trees based on their fitness

        Args:
            list_of_tree (list[Node,float]): liste contenant les arbres participant au tournoi pour former la generation suivante avec leur fitness
            landmarks_list (list[ tuple[dict[str,int], float]]) : liste des points repères sous forme de dictionnaire avec comme clés les noms des points repères sous la forme 'x1','x2','y1','y2'
            
        Returns:
            list[ list [Node,float] ]: la liste des arbres selectionnés pour former la génération suivante
        """
        if all(isinstance(item, (tuple, list)) and len(item) > 1 for item in list_of_tree):
            sorted_list = list_of_tree[:]
            sorted_list.sort(reverse=True, key=itemgetter(1))  # Create a sorted copy
            num_elites = round(self.elitism_rate * len(list_of_tree) / 100)
            return [ (Node.copy(sorted_list[k][0]),sorted_list[k][1]) for k in range(num_elites)]
        else:
            raise ValueError("All elements in list_of_tree must have at least two items.")
        

    
    
    def fit_genetic(self, desired_class: int, landmarks_list: list[dict[str, int]]) -> Node:
        """
        Génère un arbre logique à partir d'une liste de points repères reconnaissant la classe désirée.

        Args:
            desired_class (int): La classe que l'on veut que l'arbre prédise.
            landmarks_list (list[dict[str,int]]): Liste des points repères sous forme de dictionnaire.

        Returns:
            Node: Un arbre reconnaissant la classe désirée.
        """
        print("[INFO] Starting genetic algorithm...")

        # Initialisation de la population
        trees = self.generate_random_tree()
        trees = [ (tree,0) for tree in trees ]  # Initialize with fitness 0
        
        # Initialisation du meilleur global
        #best_global = None
        
        for i in range(self.number_iteration):
            # Calcul de la fitness de chaque arbre
            trees_fitness = [
                (tree[0], self.return_fitness(tree[0], desired_class, landmarks_list))
                for tree in trees
            ]

            # Mise à jour du meilleur arbre global
            #current_best = max(trees_fitness, key=lambda x: x[1])[0]
            #if best_global is None or self.return_fitness(current_best, desired_class, landmarks_list) > self.return_fitness(best_global, desired_class, landmarks_list):
            #    best_global = Node.copy(current_best)

            # Sélection des élites
            elites = self.elitism(trees_fitness, landmarks_list)

            # Sélection des parents
            parents = self.tournament_selection(trees_fitness, landmarks_list)
            # Remplace le pire parent par une copie du meilleur global
            parents.sort(key=itemgetter(1), reverse=True)
            #parents[0] = Node.copy(best_global)

            # Mutation des parents
            for tree in parents:
                if random.randint(1, 100) <= self.mutation_rate:
                    self.mutate_logic(tree[0])
                    self.mutate_values(tree[0])

            # Croisement des parents
            for j in range(0, len(parents) - 1, 2):
                self.crossover(parents[j][0], parents[j + 1][0])

            # Recrée la population complète
            total_needed = self.pop_size - (1 + len(elites) + len(parents))
            trees = elites + parents + [
                (Node.generate_tree(self.depth, self.data),0) for _ in range(total_needed)
            ]

            # Logs
            #print("Top elite fitness:", self.return_fitness(elites[0][0], desired_class, landmarks_list))
            print(f"Iteration {i + 1}/{self.number_iteration}")
            print("Taille de la population:", len(trees))

        trees.sort(key=itemgetter(1), reverse=True)
        return trees[0][0]  # Retourne l'arbre avec la meilleure fitness
    
    
    def hyperparameters_tuning(self, desired_class : int, train_data : list[dict[str,int],int],test_data : list[dict[str,int],int], hyperparameters_variation : dict[ str : (float,float,float) ], hyperpamameters_fixed : dict[ str : float]) -> list[(dict[str,int],float)] :
        """genere une règle de classification pour une classe donnée à partir d'une liste de points repères pour des hyperparamètres donnés et les ajuste pour obtenir les meilleurs hyperparamètres.
           La fonction fait varier les hyperparamètres et retourne une liste des hyperparamètres et le taux d'erreur associé à chaque combinaison d'hyperparamètres.

        Args:
            desired_class (int): la classe pour laquelle on veut générer une règle de classification
            train_data (list[dict[str,int]]): liste des points repères sous forme de dictionnaire avec comme clés les noms des points repères sous la forme 'x1','x2','y1','y2'
            test_data (list[dict[str,int]]): liste des points repères de test sous forme de dictionnaire avec comme clés les noms des points repères sous la forme 'x1','x2','y1','y2'
            hyperpamameters_variation (dict[ str : (float,float,float) ]): dictionnire de paramètres à faire varier sous la forme d'un tuple (min inclu ,max inclu,step)

        Returns:
            list[(dict[str,int],float)]: liste des hyperparamètres et le taux d'erreur associé à chaque combinaison d'hyperparamètres
        """
        results = []
        self.modify_attr(hyperpamameters_fixed)
        
        # Generate all combinations of hyperparameter values
        keys = list(hyperparameters_variation.keys())
        ranges = [MyUtils.float_range(value[0], value[1] + value[2], value[2]) for value in hyperparameters_variation.values()]
        combinations = product(*ranges)
        number_iterations = 0

        for combination in combinations:
            
            number_iterations += 1
            print(f"[INFO] Hyperparameter tuning iteration {number_iterations}")
            
            hyper_param_values = dict(zip(keys, combination))
            self.modify_attr(hyper_param_values)
            arbre = self.fit_genetic(desired_class, train_data)
            fitness = self.return_fitness(arbre, desired_class, test_data)
            relevant_attributes = {key: getattr(self, key) for key in hyperparameters_variation.keys()}
            results.append([[key, relevant_attributes[key]] for key in hyperparameters_variation.keys()] + [fitness])

        name = ""  
        for key in hyperparameters_variation.keys() :
            name += key + "_" + str(hyperparameters_variation[key][0]) + "_" + str(hyperparameters_variation[key][1]) + "_" + str(hyperparameters_variation[key][2]) + ","
            
        MyUtils.write_json_list(os.path.join(self.base_dir, "logs", "tree_hyperparameters_tuning" + name + ".json"), results)
        
        return results
    
    
    
    def modify_attr(self, modif : dict[str, int]) -> None :
        """modifie les attributs de l'objet en fonction du dictionnaire passé en paramètre

        Args:
            modify (dict[str,int]): dictionnaire contenant les attributs à modifier et leur nouvelle valeur
        """
        for key,value in modif.items() :
            setattr(self,key,value)
    
    
    
        
#    
#arbre1 = Node.generate_root_node()
#arbre1.left = Node.generate_tree(2,['x1','x2','y1','y2'])
#arbre1.right = Node.generate_tree(2,['x1','x2','y1','y2'])
#arbre2 = Node.generate_root_node()
#arbre2.left = Node.generate_tree(2,['x1','x2','y1','y2'])
#arbre2.right = Node.generate_tree(2,['x1','x2','y1','y2'])
#net = net_gen(100,100,10,2,['x1','x2','y1','y2'])
#print("arbre1 : " + arbre1.print_tree())
#print("arbre2 : " + arbre2.print_tree())
#net.crossover(arbre1,arbre2)
#print("crossovered arbre1: " + arbre1.print_tree())
#print("crossovered arbre2 : " + arbre2.print_tree())
#print(net.count_node(arbre))
#(arb) = net.get_selected_node(arbre1,2)
#print(abr)
#print(arb.print_tree())
#net.mutate_logic(arbre)
#net.mutate_values(arbre)
#print(arbre.print_tree())
#net.mutate_logic(arbre)
#net.mutate_values(arbre)
#print(arbre.print_tree())
#arbre2 = Node('AND')
#arbre2.left = Node('OR')
#arbre2.right = Node('OR')
#arbre2.left.left = Node('<')
#arbre2.left.right = Node('>')
#arbre2.left.right.right = Node('x1')
#arbre2.left.right.left = Node('y2') 
#arbre2.left.left.left = Node('x1')
#arbre2.left.left.right = Node('y2')
#arbre2.right.left = Node('>')
#arbre2.right.right = Node('<')
#arbre2.right.right.right = Node('x1')
#arbre2.right.right.left = Node('y2')
#arbre2.right.left.left = Node('x1')
#arbre2.right.left.right = Node('y2')
#print("arbre2 " + arbre2.print_tree())
#arb=net.get_selected_node(arbre2,6)
#print(arb)
#(num,ar) = net.replaceInTree(arbre,Node('a'),3)
#print("result get_selected_node " + arb.print_tree())
#print(arbre.print_tree())

