import json
import numpy as np
from utils import MyUtils
import os
class PreTraitement() :
    
    
    def __init__(self) -> None:
        self.base_dir = os.path.dirname(__file__)
        self.annotations = {}
        self.annotationsFiles = [
            os.path.join(self.base_dir,"annotations\\sample\\call.json"),
            os.path.join(self.base_dir,"annotations\\sample\\dislike.json"),
            os.path.join(self.base_dir,"annotations\\sample\\fist.json"),
            os.path.join(self.base_dir,"annotations\\sample\\four.json"),
            os.path.join(self.base_dir,"annotations\\sample\\like.json"),
            os.path.join(self.base_dir,"annotations\\sample\\mute.json"),
            os.path.join(self.base_dir,"annotations\\sample\\ok.json"),
            os.path.join(self.base_dir,"annotations\\sample\\palm.json"),
            os.path.join(self.base_dir,"annotations\\sample\\one.json"),
            os.path.join(self.base_dir,"annotations\\sample\\peace.json"),
            os.path.join(self.base_dir,"annotations\\sample\\peace_inverted.json"),
            os.path.join(self.base_dir,"annotations\\sample\\rock.json"),
            os.path.join(self.base_dir,"annotations\\sample\\stop.json"),
            os.path.join(self.base_dir,"annotations\\sample\\stop_inverted.json"),
            os.path.join(self.base_dir,"annotations\\sample\\three.json"),
            os.path.join(self.base_dir,"annotations\\sample\\three2.json"),
            os.path.join(self.base_dir,"annotations\\sample\\two_up.json"),
            os.path.join(self.base_dir,"annotations\\sample\\two_up_inverted.json"),
        ]
        self.filtered_name = {}
        self.diff_class=["call","dislike", "fist", "four", "like", "mute", "ok","palm","one","peace","peace_inverted",
                        "rock","stop","stop_inverted","three","three2","two_up","two_up_inverted"]
        self.dic_class={}
        self.data_train = []
        self.num_class = { class_name: i for i, class_name in enumerate(self.diff_class) }
        
    def generate_dict(self) : 
        """genère le dictionnaire d'annotations pour les images
        """
        # test si le fichier existe déja et l'utilise si c'est le cas
        if os.path.isfile(os.path.join(self.base_dir,"logs\\dict_name_normalize.json")) and os.access(os.path.join(self.base_dir,"logs\\dict_name_normalize.json"), os.R_OK):
            self.annotations = MyUtils.read_json(os.path.join(self.base_dir,"logs\\dict_name_normalize.json"))
        else:
            # genère le dictionnaire pour le stocker sous le format json
            dict = {}
            for source in self.annotationsFiles:
                dict.update(MyUtils.read_json(source))
            MyUtils.write_json(os.path.join(self.base_dir,"logs\\dict_name.json"),dict)
            self.normalize_all_and_save()
            self.annotations = MyUtils.read_json(os.path.join(self.base_dir,"logs\\dict_name_normalize.json"))
    
    def verification_data(self,img_id) :
        """_summary_
        Elimine les points repères n'ayant pas de labels valident pour une image donnée
        """
        l = []
        img= self.annotations[img_id]
        for i in range(len(img["labels"])):
            if img["labels"][i] == "no_gesture" :
                l.append(i)
            else :
                if img["landmarks"][i] == [] or img["bboxes"][i] == [] :
                    l.append(i)
        for i in reversed(l) :
            del img["bboxes"][i]
            del img["labels"][i]
            del img["landmarks"][i]
    
    def get_landmarks(self,img_id: str) -> list:
        """_summary_
            retourne la liste de points repères pour la 1ère mains valide et None si il n'en existe pas. 
            Il faut verifier les données avant d'utiliser cette fonction ( verification_data )
        Args:
            img_id (string): identifiant de l'image 
        """
        if self.annotations[img_id] == [] : None
        else : return self.annotations[img_id][0]
        
    
    def get_labels(self,img_id: str) -> str :
        """_summary_
        retourne le 1er label valide de l'image et None si il n'en existe pas
        Args:
            img_id (string): id de l'image
        """
        if self.annotations[img_id]["labels"] != [] :
            return self.annotations[img_id]["labels"][0]
        else : return None
    
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
        
    def calcul_distance(self,ensemble_points, point_ref : dict = { "x": 0, "y":0} ) :
        ''' distance algébrique avec 2 dimensions'''
        distance = []
        
        for point in ensemble_points :
            distance.append(   
                            np.sqrt( (point[0] - point_ref['x'])**2 + (point[1] - point_ref['y'])**2   )
                            )
        return distance
    
    def calcul_distance(self, ensemble_points , point_ref : list = [0,0]) -> list[(int)]:
        ''' distance algébrique avec 2 dimensions'''
        distance = []
        
        for point in ensemble_points :
            distance.append(   
                            np.sqrt( (point[0] - point_ref[0])**2 + (point[1] - point_ref[1])**2   )
                            )
        return distance
    
    def distance_reseau(self, id ) :
        '''retourne la distance entre les points repères selectionnées et le point repère 0
        '''
        
        landmarks = self.get_landmarks(id)
        if len(landmarks)!= 21 :
            pass
        else :
            box = self.annotations[id]["bboxes"]
            
            for k in range(21): # normalise les coordonnées par rapport a la box
                landmarks[k][0] = (landmarks[k][0]-box[0])/box[2]
                landmarks[k][1] = (landmarks[k][1]-box[1])/box[3]
            
            #ensemble_points = [ landmarks[k] for k in range(2,21,2) ] 
            #correspond aux phallanges 2,4,6,8,10,12,14,16,18,20 : voir le site de mediapipe
            ensemble_points = [ landmarks[k] for k in range(21)]
            distance = self.calcul_distance( ensemble_points, landmarks[0])
            return distance
    
    def filter_name(self) :
        """_summary_
        filtre les images en fonction de l'utilisateur et les stocke dans un fichier json
        """
        # verifie si le fichier existe et l'utilise dans ce cas 
        if os.path.isfile(os.path.join(self.base_dir,"logs\\filtered_name.json")) and os.access(os.path.join(self.base_dir,"logs\\filtered_name.json"), os.R_OK):
            self.filtered_name = MyUtils.read_json(os.path.join(self.base_dir,"logs\\filtered_name.json"))

        else:
            # réecrie le fichier et le stocke sous le format json
            for x in range(len(self.annotations)) : 
                if self.annotations[x]['user_id'] not in self.filtered_name : self.filtered_name[self.annotations[x]["user_id"]]=[x]
                else :
                    self.filtered_name[self.annotations[x]["user_id"]].append(x)
            MyUtils.write_json(os.path.join(self.base_dir,"logs\\filtered_name.json"),self.filtered_name)
              
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
        MyUtils.write_json(os.path.join(self.base_dir,"logs\\batch") + str(num_batch)+ ".json",batch)
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
        """_summary_
        retourne le vecteur de la classe correspondant au label de l'image
        Args:
            img_id (_type_): _description_

        Returns:
            : _description_
        """
        label = self.get_labels(img_id)
        if label == None :
            return None
        else : return self.diff_class[label]
             

    def convert_normalize_coo (self, img : dict ) :
        """_summary_
        change les coordonnées des poins repères en coordonnées relatives à la boxe
        Args:
            img (dict): 
        """
        landmarks = img["landmarks"]
        box = img["bboxes"]
        
        for i in range(len(landmarks)) :    
            for k in range(len(landmarks[i])): # normalise les coordonnées par rapport a la box
                img["landmarks"][i][k][0] = (landmarks[i][k][0]-box[i][0])/box[i][2]
                img["landmarks"][i][k][1] = (landmarks[i][k][1]-box[i][1])/box[i][3]
        return img

    def change_Coo_distance (self, img: dict) :
        """_summary_
        remplace les coordonnées normalizée 
        Args:
            img (dict): _description_
        """
        landmarks = img["landmarks"]
        box = img["bboxes"]
        
        for i in range(len(landmarks)) :    
            img["landmarks"][i]=(self.calcul_distance(landmarks[i], landmarks[i][0]))
        return img
    
    def change_label_vector (self , img) : 
        """_summary_
        remplace les labels par les vecteurs correspondant
        Args:
            img (dict): _description_
        """
        label = img["labels"]
        for i in range(len(label)) :
            img["labels"][i] = self.dic_class[label[i]]
        return img
    
    
    def get_data_train(self,batch) :
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        n = len(batch)
        p = round(0.8*n)
        for i in range(n) :
            img = self.annotations[batch[i]]
            try :
                img = self.change_label_vector(self.change_Coo_distance(img))
                if i<=p :
                    for j in range(len(img["landmarks"])):
                        x_train.append ([img["landmarks"][j]])
                        y_train.append([img["labels"][j]])
                else :
                    for j in range(len(img["landmarks"])):
                        x_test.append ([img["landmarks"][j]])
                        y_test.append([img["labels"][j]])
            except :
                print(batch[i],self.annotations[batch[i]]["labels"],i)
        return (np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test))
    
    
    def get_data_evolutional(self,batch : list[str]) -> list [ tuple[ dict [ str , int ], int] ] :
        """ retourne la liste des couples contenant  : - le dictionnaire de points repères pour le batch donné. 
                                                            Le dictionnaire est sous la forme { 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, ...}.
                                                       - le numéro de la classe correspondant au label de l'image

        Args:
            batch (list[str]): liste des id des images

        Returns:
           list [ tuple [ dict [ str, int ], int ] ]: la liste des couples dict,num_class
        """
        x_train = []
        l = []
        n = len(batch)
        for i in range(n) :
            img_data = self.annotations[batch[i]]
            try :
                dico = {}
                for landmarks_el in img_data["landmarks"]:
                    for j,landmark_coo in enumerate(landmarks_el) :
                        dico['x%s' %j] = landmark_coo[0]
                        dico['y%s' %j] = landmark_coo[1]
                        
                    x_train.append( (dico, self.num_class[ img_data["labels"][0]  ]) )
            except Exception as e :
                print( "error :" , e) 
            
        return (np.array(x_train))

    def get_list_coo_landmarks_name(self) -> list[str] :
        """ retourne la liste des noms des coordonnées des points repères sous la forme 'x1','x2','y1','y2'

        Returns:
            list[str]: la liste des noms des coordonnées des points repères sous la forme 'x1','x2','y1','y2'
        """
        l = []
        for i in range(21) :
            l.append("x%s" %i)
            l.append("y%s" %i)
        return l

    def normalize_all_and_save (self) :
        """Normalise les coordonnées des points repères et les sauvegarde dans un fichier json
        """
        self.annotations = MyUtils.read_json(os.path.join(self.base_dir,"logs\\dict_name.json"))
        anno = {}
        for img_id in self.annotations :
            img = self.annotations[img_id]
            self.verification_data(img_id)
            img = self.convert_normalize_coo(img)
            anno[img_id] = img
        MyUtils.write_json(os.path.join(self.base_dir,"logs\\dict_name_normalize.json",anno))
        
        
        
#preT = PreTraitement() # instantiate trainigObject
#preT.normalize_all_and_save()
#preT.generate_dict() # prepare training data (aggregate training data grouped by imageId)
#preT.create_dic_class()
#preT.filter_name()
#print(preT.get_data_train(preT.return_batch(1)))
#preT.verification_data("2d743952-faef-42c1-be40-ff5ec2f799ad")
#print(preT.annotations["2d743952-faef-42c1-be40-ff5ec2f799ad"])
#print(preT.convert_normalize_coo(preT.annotations["2d743952-faef-42c1-be40-ff5ec2f799ad"]))
#preT.verification_data("000ed71d-3661-4aa9-b38b-104d2f3013b4")
#print(preT.change_Coo_distance(preT.convert_normalize_coo(preT.annotations["000ed71d-3661-4aa9-b38b-104d2f3013b4"])))
#print(preT.annotations["00142c3d-f036-47eb-a2d9-6902a8a803e4"]) # two boxes
# print(preT.annotations["000ed71d-3661-4aa9-b38b-104d2f3013b4"]) # one box

#print(preT.change_label_vector(preT.annotations["02af3cf6-a829-43ec-b784-8db4bdd1cde8"]))
#print(preT.get_labels("f2333d37-be88-468f-91ad-45be69625002"))
#print(len(preT.annotations))
#preT.filter_name() # prepare training data (aggregate training data grouped by userId, in order to create random gesture batches)
#preT.create_dic_class() # annotation class label -> vector
#print(preT.get_true_values('01898f3e-8422-4e6a-a056-30206f905640')) # getVector
#print(preT.get_data_train(preT.return_batch(1)))

# 1 min 46 