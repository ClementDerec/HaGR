import json
import numpy as np
from scipy.interpolate import griddata

class MyUtils() :
    
    def __init__(self) -> None:
        pass
    
    def write_json(file_name,json_data : dict) :
        with open(file_name, "w") as outfile :
            json.dump(json_data,outfile)
            
    def read_json(file_name) -> dict :
        with open(file_name, "r") as infile :
            return json.load(infile)
        
    def write_json_list(file_name,json_data : list) :
        with open(file_name, "w") as outfile :
            json.dump(json_data,outfile)
            
    def read_json_list(file_name) -> list :
        with open(file_name, "r") as infile :
            return json.load(infile)
        
    def softmax(vecteur) :
        somme = sum(np.exp(vecteur))
        res =[0 for i in range(len(vecteur))]
        for i in range(len(vecteur)) :
            res[i]=np.exp(vecteur[i])/somme
        return res
    
    def float_range(start : float , stop : float, step : float):
        """generate a range of float numbers

        Args:
            start (float): _description_
            stop (float): _description_
            step (float): _description_

        Yields:
            _type_: _description_
        """
        while start < stop:
            yield start
            start += step
            
            
                    
    def find_max ( data : list) -> tuple :
        """Find the maximum value in a list of lists.

        Args:
            data (list): A list of lists containing numerical values.

        Returns:
            tuple: A tuple containing the maximum value and its index.
        """
        max_value = float('-inf')
        max_item = []
        for sublist in data:
            if sublist[-1] > max_value:
                max_value = sublist[-1]
                max_item = sublist
        return (max_item)
    
