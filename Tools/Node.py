import random

class Node:
    
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.convert_str_logic = {'>':'>', '<':'<', 'AND':'&&', 'OR':'||'}

    def evaluate(self,sample):
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
    
    def generate_tree(profondeur, data) :
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
            
        
# Example usage:
# Constructing the formula (T AND (F OR T))
root = Node('AND')
root.left = Node('T')
root.right = Node('OR')
root.right.left = Node('F')
root.right.right = Node('T')

# Evaluating the formula
result = Node.evaluate(root,['x1','x2','y1','y2'])
#print(f"The result of the formula is: {result}")

#print(Node.generate_tree(4,['x1','x2','y1','y2']).print_tree())