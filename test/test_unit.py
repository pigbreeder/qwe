import numpy as np
from src.objective import CategoricalCrossEntropy
from src.unit import *


def test_BasicUnit():
    np.random.seed(1)
    W = np.random.randn(3,2)
    X = np.random.randn(2,3)
    b = np.random.randn(2)
    dA = np.random.randn(2,2)
    print('W:',W,'\nX:',X,'\nb:',b,'\ndA:',dA)
    print('BasicUnit.forward==>\n', BasicUnit.forward(X,W,b))
    print('BasicUnit.backward==>\n', BasicUnit.backward(X,dA,W))

def test_SigmoidUnit():
    np.random.seed(1)
    X = np.random.randn(2,2)
    dA = np.random.randn(2,2)
    print('X:',X,'\ndA:',dA)
    print('SigmoidUnit.forward==>\n', SigmoidUnit.forward(X))
    print('SigmoidUnit.backward==>\n', SigmoidUnit.backward(X, dA))

def test_ReLUUnit():
    np.random.seed(1)
    X = np.random.randn(2,2)
    dA = np.random.randn(2,2)
    print('X:',X,'\ndA:',dA)
    print('ReLUUnit.forward==>\n', ReLUUnit.forward(X))
    print('ReLUUnit.backward==>\n', ReLUUnit.backward(X, dA))

def test_SoftmaxUnit():
    np.random.seed(1)
    X = np.random.randn(2,2)
    dA = np.random.randn(2,2)
    print('X:',X,'\ndA:',dA)
    print('SoftmaxUnit.forward==>\n', SoftmaxUnit.forward(X))
    print('SoftmaxUnit.backward==>\n', SoftmaxUnit.backward(X, dA))

if __name__ == '__main__':
    test_BasicUnit()
    test_SigmoidUnit()
    test_ReLUUnit()
    test_SoftmaxUnit()