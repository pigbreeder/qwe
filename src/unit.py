import numpy as np

from src.objective import CategoricalCrossEntropy

"""
计算单元
"""


class BasicUnit():
    # x.shape = (sample_size,input_size)
    # W.shape = (input_size, output_size)
    # return
    @staticmethod
    def forward(X, W=1, b=0):
        return np.add(np.dot(X, W), b)

    @staticmethod
    def backward(X, dA, W):
        # return dAA, dW, db
        dAA = np.dot(dA, W.T)
        dW = np.divide(np.dot(X.T, dA), X.shape[0])
        db = np.divide(np.sum(dA, axis=0), X.shape[0])
        return dAA, dW, db

class Unit(object):
    @staticmethod
    def forward(X):
        pass
    @staticmethod
    def backward(X, dA):
        pass
class SigmoidUnit(Unit):
    @staticmethod
    def forward(X):
        return 1.0/(1 + np.exp(-X))
    @staticmethod
    def backward(X, dA):
        s = SigmoidUnit.forward(X)
        return s * (1 - s) * dA

class ReLUUnit(Unit):
    @staticmethod
    def forward(X):
        A = np.maximum(0, X)
        return A

    @staticmethod
    def backward(X, dA):
        dZ = np.array(X,copy=True)
        dZ[X <= 0] = 0
        return dZ * dA

class SoftmaxUnit(Unit):
    @staticmethod
    def forward(X):
        return CategoricalCrossEntropy.predict(X)
    @staticmethod
    def backward(X, dA):
        return dA

_dict = {'sigmoid': SigmoidUnit,
         'ReLU': ReLUUnit,
         'softmax': SoftmaxUnit,
         }


def get(name):
    val = _dict.get(name, None)
    if val == None:
        raise Exception(('the calculate unit %s is not existed.' % name))
    return val