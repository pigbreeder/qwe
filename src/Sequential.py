from src.Activation import Activation
from src.BasicNN import BasicNN
from src.LostFunction import LostFunction
from src.Optimizer import Optimizer
import numpy as np

class Sequential(object):
    def __init__(self, layers=None):
        self.layers = []
        for layer in layers:
            self.add(layer)

    def add(self, layer):
        if self.layers == []:
            if layer.input_size == 0:
                raise Exception('must define input data size')
        else:
            last_layer = self.layers[-1]
            layer.set_param(last_layer.node_size)
        self.layers.append(layer)

    def compile(self, loss='cross_entropy', optimizer='sgd'):
        self.loss = LostFunction.cross_entropy
        self.loss_der = LostFunction.cross_entropy_derivative
        self.optimizer = Optimizer.SGD

    def calc_lost(self, y_label):
        return self.loss(self.layers[-1].data_forward, y_label)

    def fit(self,x_train, y_train, epochs=10, batch_size=32):
        pass
        self.optimizer(x_train,y_train,self)

    def predict(self, x_test):
        pass

    def forward(self, input_data):
        input_data_ = input_data
        for layer in self.layers:
            input_data = layer.forward(input_data)


    def backward(self,y_train, learning_rate):
        pass
        m = y_train.shape[0]
        dA = self.loss_der(self.layers[-1].data_forward, y_train)
        layer_len = len(self.layers)
        last_W = self.layers[-1].W
        for idx in range(layer_len-2,-1,-1):
            cur_layer = self.layers[idx]
            grad_W,grad_b = cur_layer.backward(dA,last_W)
            last_W = np.copy(cur_layer.W)
            cur_layer.W = cur_layer.W - learning_rate * (1./m)* np.sum(grad_W,axis=0)
            cur_layer.b = cur_layer.b - learning_rate * (1./m)* np.sum(grad_b,axis=0)

if __name__ == '__main__':
    import os
    print(os.getcwd())
    seq = Sequential([BasicNN(1,input_size=4),Activation('sigmoid')])
    seq.compile()
    np.random.seed(10)
    x_train = np.random.rand(10,4)
    y_train = np.dot(x_train, np.array([1,2,3,4])) + 0.5
    seq.fit(x_train, y_train)
