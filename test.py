from utils import MyUtils
from algo_gen_fully_corrected import net_gen
from Data.pre_traitement import PreTraitement
from Tools.plot import Plot

preT = PreTraitement()
preT.generate_dict()
preT.filter_name()

def fit_try ():
    train_data = preT.get_data_evolutional(preT.return_batch(1))
    test_data = preT.get_data_evolutional(preT.return_batch(2))
    x_train = [ train_data[i] for i in range(10000)]
    x_test = [ test_data[i] for i in range(10000)]
    data = preT.get_list_coo_landmarks_name()
    reseau = net_gen(10.40,71.5,50,6,data,2,55,60,33,3)
    #arbre = reseau.fit_genetic(1,x_train)
    return(reseau.hyperparameters_tuning(1,x_train,x_test,{"mutation_rate" : [1,40,5], "elitism_rate" : [1,40,5]},{}))
    #return arbre
    
data = Plot.get_plot_coordinates_from_list(fit_try())
Plot.plot_smoothed_surface_with_max(data[0],data[1],data[2], "Fitness en fonction du taux de mutation et du taux d'élitisme","Taux de mutation ( en % )","Taux d'élitisme (en %)","Fitness",0)
