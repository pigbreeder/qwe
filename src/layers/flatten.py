from src.layers.layer import Layer
from src.unit import BasicUnit
import src.initialization
import numpy as np
class Flatten(Layer):
    def __init__(self, input_size=(0,), name=''):
        if not isinstance(input_size, (list, tuple)):
            raise Exception('please use list or tuple type to set input_size')
        super().__init__(0, input_size, name)


    def build(self, loc_idx, input_size, init_param_method):
        super().build(loc_idx, input_size, init_param_method)
        self.output_size = np.product(input_size)

    def forward(self, X):
        return X.reshape(X.shape[0], self.output_size)
    def backward(self, X, dA):
        return dA.reshape((dA.shape[0],) + self.input_size), None, None

if __name__ == '__main__':
    l = Flatten(input_size=(3,4,4),name='test')
    l.build(0,(3,4,4),'random')
    X = np.random.randn(2,3,4,4)
    Y = l.forward(X)
    print(X.shape,Y.shape)
    Y=l.backward(X,np.random.randn(2,48))
