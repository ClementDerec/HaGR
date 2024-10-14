import json
import numpy as np
from utils import MyUtils
import os
class PreTraitement() :
    
    
    def __init__(self) -> None:
        #pass
        self.annotations = {}
        self.annotationsFiles = [
            "C:\\DATA\\DEV\\TIPE\\HaGR\\annotations\\sample\\call.json",
            "C:\\DATA\\DEV\\TIPE\\HaGR\\annotations\\sample\\dislike.json",
            "C:\\DATA\\DEV\\TIPE\\HaGR\\annotations\\sample\\fist.json",
            "C:\\DATA\\DEV\\TIPE\\HaGR\\annotations\\sample\\four.json",
            "C:\\DATA\\DEV\\TIPE\\HaGR\\annotations\\sample\\like.json",
            "C:\\DATA\\DEV\\TIPE\\HaGR\\annotations\\sample\\mute.json",
            "C:\\DATA\\DEV\\TIPE\\HaGR\\annotations\\sample\\ok.json",
            "C:\\DATA\\DEV\\TIPE\\HaGR\\annotations\\sample\\palm.json",
            "C:\\DATA\\DEV\\TIPE\\HaGR\\annotations\\sample\\one.json",
            "C:\\DATA\\DEV\\TIPE\\HaGR\\annotations\\sample\\peace.json",
            "C:\\DATA\\DEV\\TIPE\\HaGR\\annotations\\sample\\peace_inverted.json",
            "C:\\DATA\\DEV\\TIPE\\HaGR\\annotations\\sample\\rock.json",
            "C:\\DATA\\DEV\\TIPE\\HaGR\\annotations\\sample\\stop.json",
            "C:\\DATA\\DEV\\TIPE\\HaGR\\annotations\\sample\\stop_inverted.json",
            "C:\\DATA\\DEV\\TIPE\\HaGR\\annotations\\sample\\three.json",
            "C:\\DATA\\DEV\\TIPE\\HaGR\\annotations\\sample\\three2.json",
            "C:\\DATA\\DEV\\TIPE\\HaGR\\annotations\\sample\\two_up.json",
            "C:\\DATA\\DEV\\TIPE\\HaGR\\annotations\\sample\\two_up_inverted.json",
        ]
        self.filtered_name = {}
        self.diff_class=["call","dislike", "fist", "four", "like", "mute", "ok","palm","one","peace","peace_inverted","rock","stop","stop_inverted","three","three2","two_up","two_up_inverted"]
        self.dic_class={}
        self.data_train = []
        
    def generate_dict(self) : 
        # test si le fichier existe déja et l'utilise si c'est le cas
        if os.path.isfile("C:\\DATA\\DEV\\TIPE\\HaGR\\logs\\dict_name.json") and os.access("C:\\DATA\\DEV\\TIPE\\HaGR\\logs\\dict_name.json", os.R_OK):
            self.annotations = MyUtils.read_json("C:\\DATA\\DEV\\TIPE\\HaGR\\logs\\dict_name.json")
        else:
            # genère le dictionnaire pour le stocker sous le format json
            dict = {}
            for source in self.annotationsFiles:
                dict.update(MyUtils.read_json(source))
            MyUtils.write_json("C:\\DATA\\DEV\\TIPE\\HaGR\\logs\\dict_name.json",dict)
            self.annotations = dict
    
    def get_landmarks(self,id) :
        labels = self.get_labels(id)
        for i in range(len(labels)) :
            if labels[i] != "no_gesture" :
                self.annotations[id]["bboxes"]= self.annotations[id]["bboxes"][i] # selectionne les boxes et coo pour les mains avec des labels valident
                return self.annotations[id]['landmarks'][i]
        return []
    
    def get_labels(self,id) :
        return(self.annotations[id]['labels'])
    
    def calcul_barycentre(self,landmarks) :
        barycentre = {"x":0,"y":0}
        i=0
        for point in landmarks :
            barycentre['x'] += point[0]
            barycentre['y'] += point[1]
            i += 1
        barycentre['x'] = barycentre['x']/i
        barycentre['y'] = barycentre['y']/i

        return(barycentre)
        
    def calcul_distance(self,ensemble_points, point_ref : dict) :
        ''' distance algébrique avec 2 dimensions'''
        distance = []
        
        for point in ensemble_points :
            distance.append(   
                            np.sqrt( (point[0] - point_ref['x'])**2 + (point[1] - point_ref['y'])**2   )
                            )
        return distance
    
    def distance_reseau(self, id ) :
        '''retourne la distance entre les points repères selectionnées et le barycentre
        '''
        
        landmarks = self.get_landmarks(id)
        if len(landmarks)!= 21 :
            pass
        else :
            box = self.annotations[id]["bboxes"]
            
            for k in range(21): # normalise les coordonnées par rapport a la box
                landmarks[k][0] = (landmarks[k][0]-box[0])/box[2]
                landmarks[k][1] = (landmarks[k][1]-box[1])/box[3]
            
            #ensemble_points = [ landmarks[k] for k in range(2,21,2) ] # correspond aux phallanges 2,4,6,8,10,12,14,16,18,20 : voir le site de mediapipe
            ensemble_points = [ landmarks[k] for k in range(21)]
            distance = self.calcul_distance( ensemble_points, self.calcul_barycentre(ensemble_points))
            return distance
    
    def filter_name(self) :
        # verifie si le fichier existe et l'utilise dans ce cas 
        if os.path.isfile("C:\\DATA\\DEV\\TIPE\\HaGR\\logs\\filtered_name.json") and os.access("C:\\Users\\cleme\\OneDrive\\Documents\\TIPE\\Dev_NN\\logs\\filtered_name.json", os.R_OK):
            self.filtered_name = MyUtils.read_json("C:\\DATA\\DEV\\TIPE\\HaGR\\logs\\filtered_name.json")

        else:
            # réecrie le fichier et le stocke sous le format json
            for x in self.annotations : 
                if self.annotations[x]['user_id'] not in self.filtered_name : self.filtered_name[self.annotations[x]["user_id"]]=[x]
                else :
                    self.filtered_name[self.annotations[x]["user_id"]].append(x)
            MyUtils.write_json("C:\\DATA\\DEV\\TIPE\\HaGR\\logs\\filtered_name.json",self.filtered_name)
              
    def return_batch(self,num_batch) : 
        batch = []
        k=0
        if num_batch == -1 :
            for user_id in self.filtered_name :
                batch += self.filtered_name[user_id] 
        else :
            for user_id in self.filtered_name :
                if k>=(num_batch*100) and k<(num_batch+1)*100 :
                    batch += self.filtered_name[user_id] 
                elif k>=(num_batch+1)*100:
                    break
                k+=1        
        MyUtils.write_json("C:\\DATA\\DEV\\TIPE\\HaGR\\logs\\batch" + str(num_batch)+ ".json",batch)
        return batch
    
    def create_dic_class(self) :
        """genere les vecteurs utilisés pour calculer les erreurs"""
        l = len(self.diff_class)
        for i in range(l) :
            self.dic_class[self.diff_class[i]] = [0 for j in range(i)]      
            self.dic_class[self.diff_class[i]].append(1)
            for j in range(l-1-i) : 
                self.dic_class[self.diff_class[i]].append(0)
    
    def get_true_values(self,img_id):
        label = self.get_labels(img_id)
        for j in label :
            if j == 'no_gesture' :
                continue
            else :
                return self.dic_class[j]

    def new_method(self):
        pass


        # 
        # 
        
        {
            
            
            
            
            
        }



    def get_data_train(self,batch) :
        x_train = []
        y_train = []
        for img_id in batch :
            #landmarks = self.get_landmarks(img_id)
            distance = self.distance_reseau(img_id)
            label = self.get_true_values(img_id)
            if label ==  None :
                print("no label")
                continue
            elif distance == None :
                continue 
            else :
                x_train.append ([distance])
                y_train.append([label])
        return (np.array(x_train),np.array(y_train))
    
#preT = PreTraitement() # instantiate trainigObject
#preT.generate_dict() # prepare training data (aggregate training data grouped by imageId)
#print(preT.get_labels("f2333d37-be88-468f-91ad-45be69625002"))
#print(len(preT.annotations))
#preT.filter_name() # prepare training data (aggregate training data grouped by userId, in order to create random gesture batches)
#preT.create_dic_class() # annotation class label -> vector
#print(preT.get_true_values('01898f3e-8422-4e6a-a056-30206f905640')) # getVector
#print(preT.get_data_train(preT.return_batch(1)))

# 1 min 46 