from src.layers.layer import Layer
from src.unit import BasicUnit
import src.initialization
import numpy as np
class Dense(Layer):
    def __init__(self, node_size, input_size=0, name=''):
        super().__init__(node_size, input_size, name)

    def build(self, loc_idx, input_size, init_param_method):
        super().build(loc_idx, input_size, init_param_method)
        self.W = src.initialization.get(init_param_method)((self.input_size, self.output_size))
        self.b = np.zeros((1, self.output_size))

    def forward(self, X):
        return BasicUnit.forward(X, self.W, self.b)
    def backward(self, X, dA):
        return BasicUnit.backward(X, dA, self.W)

if __name__ == '__main__':
    l = Dense(3,name='test')
    l.intro()
    l.describe()