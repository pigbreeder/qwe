import numpy as np

from src.layers.layer import Layer
import src.unit

class Activation(Layer):
    def __init__(self, input_size, activation_method='', output_size=0, loc_idx=-1, name=''):
        self.activation = src.unit.get(activation_method)
        if self.activation == None:
            raise Exception('Wrong activation method, please check it!')
        super().__init__(output_size,input_size, name)

    def forward(self, X):
        return self.activation.forward(X)
    def backward(self, X, dA):
        return self.activation.backward(X, dA)
if __name__ == '__main__':
    a =Activation(3,'sigmoid')
    src.unit.SoftmaxUnit.forward(1)