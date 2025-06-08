import random

class Node:

    
    def __init__(self, value : str) :
        self.value : str = value 
        self.left :Node = None 
        self.right :Node = None
        self.convert_str_logic = {'>':'>', '<':'<', 'AND':'&&', 'OR':'||'}

    def evaluate(self,sample: dict[str:float] ) -> bool:
        """_summary_

        Args:
            sample (dict[ str : float ]): dictionnaire ayant pour clés les noms des coordonnées des points repères et pour clés les valeurs associées

        Returns:
            bool : valeur de la formule logique
        """
        if self.value == 'T':
            return True
        elif self.value == 'F':
            return False
        elif self.value == 'NOT':
            return not Node.evaluate(self.left,sample)
        elif self.value == 'AND':
            return Node.evaluate(self.left,sample) and Node.evaluate(self.right,sample)
        elif self.value == 'OR':
            return Node.evaluate(self.left,sample) or Node.evaluate(self.right,sample)
        elif self.value == '>':
            return Node.evaluate(self.left,sample) > Node.evaluate(self.right,sample)
        elif self.value == '<' :
            return Node.evaluate(self.left,sample) < Node.evaluate(self.right,sample)
        else :
            return sample[self.value]
        
    def generate_root_node() :
        k = random.randint(1,2)
        if k==1 :
            tree = Node('AND')
        else :
            tree = Node('OR')
        return tree
    
    def generate_tree(profondeur : int , data : list[str])  -> 'Node':
        """_summary_

        Args:
            profondeur (int): profondeur max de l'arbre
            data ( list[ str ] ): liste des noms des coordonnées des points repères sous la forme 'x1','x2','y1','y2'

        Returns:
            Node: abre représentant une formule logique générée aléatoirement
        """
        # max depth is reached
        if profondeur == 0 :
            j = random.randint(0,1)
            if j==1 :
                tree = Node('>')
                i = random.randint(0,len(data)-1) 
                l = random.randint(0,len(data)-1)
                tree.left = Node(data[i])
                tree.right = Node(data[l]) 
            else :
                tree = Node('<')
                i = random.randint(0,len(data)-1) 
                l = random.randint(0,len(data)-1)
                tree.left = Node(data[i])
                tree.right = Node(data[l]) 
        # max depth is not reached
        else :
            k = random.randint(0,2)
            if k==1 :
                tree = Node('AND')
                tree.left = Node.generate_tree(profondeur -1,data)
                tree.right = Node.generate_tree(profondeur -1,data)
            elif k==2 :
                tree = Node('OR')
                tree.left = Node.generate_tree(profondeur -1,data)
                tree.right = Node.generate_tree(profondeur -1,data)
            else :
                j = random.randint(0,1)
                if j==1 :
                    tree = Node('>')
                    i = random.randint(0,len(data)-1) 
                    l = random.randint(0,len(data)-1)
                    tree.left = Node(data[i])
                    tree.right = Node(data[l]) 
                else :
                    tree = Node('<')
                    i = random.randint(0,len(data)-1) 
                    l = random.randint(0,len(data)-1)
                    tree.left = Node(data[i])
                    tree.right = Node(data[l]) 
        return tree
    
    def print_tree (self) :
        if self != None :
            if self.value in self.convert_str_logic :
                # node
                return("(" + Node.print_tree(self.left)+" " + self.convert_str_logic[self.value]+" " + Node.print_tree(self.right) + ")")
            else :
                # leaf
                return(Node.print_tree(self.left)+self.value+Node.print_tree(self.right))
        else :
            return ""
    
    def copy(tree_node : 'Node') :
        if tree_node != None :
            arbre = Node(tree_node.value)
            arbre.left = Node.copy(tree_node.left)
            arbre.right = Node.copy(tree_node.right)
            return arbre
        else :
            return None        
        
# Example usage:
# Constructing the formula (T AND (F OR T))
#root = Node('AND')
#root.left = Node('T')
#root.right = Node('OR')
#root.right.left = Node('F')
#root.right.right = Node('T')
#root.print_tree()
# Evaluating the formula
#result = Node.evaluate(root,['x1','x2','y1','y2'])
#print(f"The result of the formula is: {result}")

#print(Node.generate_tree(4,['x1','x2','y1','y2']).print_tree())