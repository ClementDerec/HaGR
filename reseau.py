import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_derivee
from losses import mse, mse_derivee
from pre_traitement import PreTraitement
from utils import MyUtils
preT = PreTraitement()
preT.generate_dict()
preT.filter_name()
preT.create_dic_class()


(x_train,y_train) = preT.get_data_train(preT.return_batch(-1))

net = Network()
net.add(FCLayer(21,100))
net.add(ActivationLayer(tanh,tanh_derivee))
net.add(FCLayer(100,50))
net.add(ActivationLayer(tanh,tanh_derivee))
net.add(FCLayer(50,18))
net.add(ActivationLayer(tanh,tanh_derivee))

net.use(mse,mse_derivee)
net.fit(x_train[:3000],y_train[0:3000], epochs=3, learning_rate=0.1) # il faut fournir une matrice 1,10
# a la place du tableau de distance 10, 

# test on 3 samples
out = net.predict(x_train[0:3])
print("\n")
print("predicted values : ")
print(MyUtils.softmax(out[0][0]),MyUtils.softmax(out[1][0]),MyUtils.softmax(out[2][0]), end="\n")
print("true values : ")
print(y_train[0:3])
