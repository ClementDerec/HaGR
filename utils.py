import json
import numpy as np

class MyUtils() :
    
    def __init__(self) -> None:
        pass
    
    def write_json(file_name,json_data) :
        with open(file_name, "w") as outfile :
            json.dump(json_data,outfile)
            
    def read_json(file_name) :
        with open(file_name, "r") as infile :
            return json.load(infile)
        
    def softmax(vecteur) :
        somme = sum(np.exp(vecteur))
        res =[0 for i in range(len(vecteur))]
        for i in range(len(vecteur)) :
            res[i]=np.exp(vecteur[i])/somme
        return res
    
        