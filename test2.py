
from Tools.plot import Plot
import numpy as np

def fit_try ():
    data = Plot.get_plot_coordinates_from_json("d://CLEMENT//logs//tree_hyperparameters_tuningpop_size_10_90_10,.json")
    Plot.plot_1d_surface(data[0], data[1], title="Fitness en fonction du nombre d'individus",
                        xlabel="Nombre d'individus", ylabel="Fitness")
    
fit_try()

def adazd():
    data1 = Plot.get_plot_coordinates_from_json("d://CLEMENT//logs//RNN_hyperparameters_tuningepochs_24_24_1,learning_rate_0.0001_0.0015_0.0001,.json")
    data2 = Plot.get_plot_coordinates_from_json("d://CLEMENT//logs//RNN_hyperparameters_tuningepochs_24_24_1,learning_rate_0.0016_0.003_0.0001,.json")
    data3 = Plot.get_plot_coordinates_from_json("d://CLEMENT//logs//RNN_hyperparameters_tuningepochs_24_24_1,learning_rate_0.0031_0.0045_0.0001,.json")
    data4 = Plot.get_plot_coordinates_from_json("d://CLEMENT//logs//RNN_hyperparameters_tuningepochs_24_24_1,learning_rate_0.0046_0.01_0.0001,.json")
    data5 = Plot.get_plot_coordinates_from_json("d://CLEMENT//logs//RNN_hyperparameters_tuningepochs_24_24_1,learning_rate_0.01_0.1_0.005,.json")
    data = [[],[]]
    data = [data1[1] + data2[1] + data3[1] + data4[1] + data5[1], data1[2] + data2[2] + data3[2] + data4[2] + data5[2]]
    data[0] = np.array(data[0])

    # Find the index of the minimum value in data[1]
    min_index = np.argmin(np.array(data[1]))
    # Print the value of data[0] corresponding to the minimum value of data[1]
    print("Value of data[0] at min(data[1]):", data[0][min_index])
    #Plot.plot_1d_surface(data[0], np.array(data[1]), title = "Erreur en fonction du taux d'apprentissage", xlabel = "Taux d'apprentissage (en pourcentage)", ylabel = "Erreur")
    
#adazd()