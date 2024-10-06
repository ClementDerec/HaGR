import matplotlib.pyplot
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

#img = mp.Image.create_from_file("C:\\Users\\cleme\\OneDrive\\Documents\\TIPE\\Dev_NN\\main1.jpeg")
img = mp.Image.create_from_file("C:\\Users\\cleme\\OneDrive\\Documents\\TIPE\\Dev_NN\\Images\\subsample\\three2\\0a77d5fe-3396-43a2-9feb-3e5822bba4e6.jpg")
# obient les coordonnées des points repères en utilisant le modèle de mediapipe 

def get_landmarks(img):
    # Create an HandLandmarker object.
    base_options = python.BaseOptions(model_asset_path='C:\\Users\\cleme\\OneDrive\\Documents\\TIPE\\Dev_NN\\hand_landmarker.task')
    
    options = vision.HandLandmarkerOptions(base_options=base_options,num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # Detect hand landmarks from the input image.
    detection_result = detector.detect(img)
    # hand_landmarks[0] sont les points repères normalisé
    landmarks = detection_result.hand_landmarks[0]
    
    return(landmarks)
print(get_landmarks(img))
    
# met en forme les données pour le réseau de neurones
# calcul du barycentre  

def calcul_barycentre(landmarks) :
    barycentre = {"x":0,"y":0,"z":0}
    i=0
    for point in landmarks :
        barycentre['x'] += point.x
        barycentre['y'] += point.y
        barycentre['z'] += point.z
        i += 1
    barycentre['x'] = barycentre['x']/i
    barycentre['y'] = barycentre['y']/i
    barycentre['z'] = barycentre['z']/i

    return(barycentre)


def calcul_distance(ensemble_points, point_ref : dict) :
    ''' distance algébrique avec 3 dimensions'''
    distance = []
    for point in ensemble_points :
        distance.append(   
                        np.sqrt( (point.x - point_ref['x'])**2 + (point.y - point_ref['y'])**2 + (point.z - point_ref['z'])**2  )
                         )
    return distance

def distance_reseau( landmarks ) :
    '''retourne la distance entre les points repères selectionnées et le barycentre,
        pour obtenir des ditance dépendant peu des tailles de mains, on les divise par la distance poignet-index mcp
    '''
    ensemble_points = [ landmarks[k] for k in range(2,21,2)]
    ref = calcul_distance( ensemble_points=[landmarks[0]], point_ref={"x":landmarks[5].x , "y":landmarks[5].y, "z":landmarks[5].z})
    distance = calcul_distance( ensemble_points, calcul_barycentre(landmarks))
    for i in range(len(distance)) :
        distance[i] = distance[i]/ref[0]
    return distance


def forward_propagation (x, w_n , w_b ) :
    
    y = w_b # initiallisé avec le neurone de biais
    for i in range(len(x)) :  
        y += x[i]*w_n[i] 
        
    return np.tanh(y) # utilse tanh comme fonction d'activation

def loose_function( predicted_values, real_values) :
    '''la fonction de perte utilisee ici est la MSE (mean squared error)'''
    n = len(predicted_values)
    error = 0
    for i in range(n) : 
        error += ( real_values[i] - predicted_values[i])**2
    return error/n

def last_gradient ( y,delta,i,j, gradient , predicted_values,real_values ) :
    '''calcule la dérivée partielle de la fonction d'erreur selon le poids w(i,j).
    Ce calcule est valable pour les poids menant aux neurones de sortie'''
    n = len(real_values)
    delta[j] = (2/n)*(predicted_values[j]-real_values[j])*(1-y[j]**2)
    gradient[i][j] = y[i]*delta[j]

#print(distance_reseau(get_landmarks(img)))


def softmax(X) :
    somme = 0
    l=[]
    for x in X :
        l.append ( np.exp(x))
        somme+= np.exp(x)
    return l/somme

def hidden_gradient(y,sigma,delta,i,j,gradient,w, eta):
    '''calcul la dérivéé partielle de la fonction d'erreur par le poids w(i,j) 
       et l'ajoute a la matrice gradient.
       delta et y sont des données déja complètes.
       sigma est créé au fur et a mesure et gradient aussi.
       eta est le coefficient d'apprentissage
    '''
    for k in range(len(sigma[j+1])) : # sigma[j+1] est en bijection avec dest(j)
        delta[j] += delta[k]*w[j][k]*(1-y[j]**2)
         
    gradient[i][j]  = y[i]*delta[j] # derivee partielle de la fonction d'erreur suivant le poids w(i,j)
    
    # on modifie les poids allant de j à la couche suivante (les neurones k)
    for k in range(len(delta[j+1])) :
        w[j][k] = w[j][k] -eta*gradient[j][k]  # correspond a la descente de gradient
        
def descente_gradient(w,delta,sigma,y,real_values,predicted_values,gradient,eta) :
    pass

def softmax( X ) :
    vecteur = []
    somme = 0
    for x in X :
        vecteur.append(np.exp(x))
        somme += np.exp(x)
    return vecteur/somme
