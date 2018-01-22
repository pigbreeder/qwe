from src.layers.layer import Layer
from src.unit import BasicUnit
import src.initialization
import numpy as np
class Dropout(Layer):
    def __init__(self,keep_prob, input_size=(0,), name=''):
        super().__init__(0, input_size, name)
        self.keep_prob = keep_prob

    def build(self, loc_idx, input_size, init_param_method):
        super().build(loc_idx, input_size, init_param_method)
        self.node_size = self.output_size = self.input_size

    def forward(self, X):
        drop = np.random.rand(*X.shape)
        self.drop_mask = drop < self.keep_prob
        X[self.drop_mask] = 0
        X /= self.keep_prob
        return X
    def backward(self, X, dA):
        dA[self.drop_mask] = 0
        dA /= self.keep_prob
        return dA, None, None

if __name__ == '__main__':
    l = Dropout(3,(3,4,4),name='test')
    X = np.random.randn(2,3,4,4)
    Y = l.forward(X)
    print(X.shape,Y.shape)
    Y=l.backward(X,np.random.randn(2,3,4,4))
