import numpy as np

from src.layers.layer import Layer
import src.unit

class Activation(Layer):
    def __init__(self, activation_method, node_size=0, input_size=0, loc_idx=-1, name=''):
        self.activation = src.unit.get(activation_method)
        if self.activation == None:
            raise Exception('Wrong activation method, please check it!')
        super().__init__(node_size,input_size, name)

    def build(self, loc_idx, input_size, init_param_method):
        super().build(loc_idx, input_size, init_param_method)
        self.node_size = self.output_size = input_size

    def forward(self, X):
        return self.activation.forward(X)
    def backward(self, X, dA):
        return self.activation.backward(X, dA), None, None
if __name__ == '__main__':
    a =Activation('sigmoid', 3)
