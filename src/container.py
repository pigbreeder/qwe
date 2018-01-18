from src.layers import *
import src.optimizer
import src.objective
import numpy as np


class Container(object):
    print('To check the detail parameter of layer,you can use describe func.')


class Sequential(Container):
    def __init__(self, layers=None):
        self.layers = []
        for layer in layers:
            self.add(layer)

    def add(self, layer):
        if not self.layers:
            if layer.input_size == 0:
                raise Exception('must define input data size')
        self.layers.append(layer)

    def compile(self, loss, optimizer='sgd', init_param_method='random'):
        self.loss = src.objective.get(loss)
        self.optimizer = src.optimizer.get(optimizer)
        self.init_param_method = init_param_method
        input_size = self.layers[0].input_size
        for idx, layer in enumerate(self.layers):
            layer.build(idx, input_size, init_param_method)
            input_size = layer.input_size


    def calc_lost(self, x_train, y_label):
        return self.loss(self.predict(x_train), y_label)

    def fit(self, x_train, y_train, epochs=10, learning_rate=0.01, batch_size=32):
        self.optimizer(x_train, y_train, self, epochs, learning_rate, batch_size)

    def predict(self, x_test):
        return self.forward(x_test)

    def forward(self, input_data):
        input_data_ = input_data
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def backward(self, y_train, learning_rate):
        pass


if __name__ == '__main__':
    pass
