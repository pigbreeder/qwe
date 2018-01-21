from src.layers.layer import Layer
from src.unit import BasicUnit
import src.initialization
import numpy as np
class Flatten(Layer):
    def __init__(self, node_size, input_size=(0,), name=''):
        if not isinstance(input_size, (list, tuple)):
            raise Exception('please use list or tuple type to set input_size')
        super().__init__(node_size, input_size, name)
        self.output_size = np.product(self.input_size[:])

    def forward(self, X):
        return X.reshape(X.shape[0] * self.output_size)
    def backward(self, X, dA):
        return dA.reshape((dA.shape[0],) + self.input_size)

if __name__ == '__main__':
    l = Flatten(3,(3,4,4),name='test')
    X = np.random.randn(2,3,4,4)
    Y = l.forward(X)
    print(X,Y)
    Y=l.backward(X,np.random.randn(2,48))
    print(Y)
