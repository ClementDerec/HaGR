import numpy as np
from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_derivee,relu,relu_derivee, sigmoid,sigmoid_derivee, softmax, softmax_derivee
from losses import mse, mse_derivee, cross_entropy, cross_entropy_derivee
from pre_traitement import PreTraitement
from utils import MyUtils
from plot import Plot
import losses
preT = PreTraitement()

preT.generate_dict()
preT.filter_name()
preT.create_dic_class()

(x_train,y_train,x_test,y_test) = preT.get_data_train(preT.return_batch(-1))
print(len(x_train) + len(x_test))
print(len(y_train) + len(y_test))
net = Network()
net.add(FCLayer(21,100))
net.add(ActivationLayer(tanh,tanh_derivee))
net.add(FCLayer(100,200))
net.add(ActivationLayer(tanh,tanh_derivee))
net.add(FCLayer(200,100))
net.add(ActivationLayer(tanh,tanh_derivee))
net.add(FCLayer(100,50))
net.add(ActivationLayer(tanh,tanh_derivee))
net.add(FCLayer(50,18))
net.add(ActivationLayer(tanh,tanh_derivee))
from softmax_layer import SoftmaxLayer  # Ensure you have a dedicated SoftmaxLayer implemented

net.add(SoftmaxLayer(18,18))

net.use(cross_entropy, cross_entropy_derivee)
#data = net.hyperparameters_tuning(x_train,y_train,x_test,y_test,{},{"epochs" : [24,24,1], "learning_rate" : [0.01,0.10,0.005]})
#data = Plot.get_plot_coordinates_from_list(data)
#Plot.plot_1d_surface(data[1],data[2], title = "2D surface", xlabel = "epochs", ylabel = "learning_rate")
net.fit(x_train,y_train) # il faut fournir une matrice 1,10
net.test(x_test,y_test)
# a la place du tableau de distance 10, 

# test on 3 samples
out = net.predict(x_test[110:113])
print("\n")
print("predicted values : ")
#rint(MyUtils.softmax(out[0][0]),MyUtils.softmax(out[1][0]),MyUtils.softmax(out[2][0]), end="\n")
print("true values : ")
print(y_test[110:113])



