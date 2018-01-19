from src.layers import *
import src.optimizer
import src.objective
import numpy as np


class Container(object):
    # print('To check the detail parameter of layer,you can use describe func.')
    pass

class Sequential(Container):
    def __init__(self, layers=None):
        self.layers = []
        if not layers:
            return
        for layer in layers:
            self.add(layer)

    def add(self, layer):
        if not self.layers:
            if layer.input_size == 0:
                raise Exception('must define input data size')
        self.layers.append(layer)

    def compile(self, objective, optimizer='bgd', init_param_method='uniform'):
        self.objective = src.objective.get(objective)
        self.optimizer = src.optimizer.get(optimizer)(self)
        self.init_param_method = init_param_method
        input_size = self.layers[0].input_size

        for idx, layer in enumerate(self.layers):
            layer.build(idx, input_size, init_param_method)
            input_size = layer.output_size

    def evaluate_loss(self, x_train, y_label):
        return self.objective.loss(self.predict(x_train), y_label)

    def fit(self, x_train, y_train, epochs=1000, learning_rate=0.01, batch_size=32):
        self.optimizer.set_param(n_epoch=epochs, learning_rate=learning_rate, batch_size=batch_size)
        self.optimizer.iterate(x_train,y_train)

    def predict(self, x_test):
        return self.forward(x_test)

    def forward(self, input_data):
        self.input_data = [input_data]
        for layer in self.layers:
            input_data = layer.forward(input_data)
            self.input_data.append(input_data)
        return input_data

    def backward(self, y_predict, y_train):
        dA = self.objective.diff(y_predict, y_train)
        layer_len = len(self.layers)
        grads = []
        params = []
        for idx in range(layer_len - 1, -1, -1):
            cur_layer = self.layers[idx]
            dA, grad_W, grad_b = cur_layer.backward(self.input_data[idx], dA)
            if grad_W is not None:
                grads.append((grad_W, grad_b))
                params.append((cur_layer.W, cur_layer.b))
        return params, grads
    def describe(self):
        for layer in self.layers:
            layer.describe()

if __name__ == '__main__':
    pass
