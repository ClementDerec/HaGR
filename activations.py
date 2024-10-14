import numpy as np

# fonction d'activation et sa dérivée, agisse sur un vecteur
def tanh(x):
    return np.tanh(x)

def tanh_derivee(x):
    return 1-np.tanh(x)**2

def relu(x) :
    output = []
    for y in x :
        if y<0 : output.append(0)
        else : output.append(y)
    return np.array(output) 
   
def relu_derivee(x):
    output = []
    for y in x :
        if y<=0 : output.append(0)
        else : output.append(1)
    return np.array(output)

print(relu_derivee([1,3,2,-3,-7,8,0.12]))