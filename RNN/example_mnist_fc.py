import numpy as np

from network import Network
from RNN.fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_derivee
from losses import mse, mse_derivee

from keras.datasets import mnist
from keras.utils import to_categorical

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# training data : 60000 echantillons
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = to_categorical(y_train)
print("y_train: " , y_train)
print("x_train: " , x_train)
# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = to_categorical(y_test)

# Réseau
net = Network()
net.add(FCLayer(28*28, 100))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add(ActivationLayer(tanh, tanh_derivee))
net.add(FCLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(tanh, tanh_derivee))
net.add(FCLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(ActivationLayer(tanh, tanh_derivee))

# s'entraine sur 1000 echantillons

net.use(mse, mse_derivee)
net.fit(x_train[0:1000], y_train[0:1000], epochs=35, learning_rate=0.1)

# test sur 3 echantillons
out = net.predict(x_test[0:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:3])
