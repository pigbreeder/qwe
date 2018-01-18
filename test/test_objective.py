import numpy as np
from src.objective import CategoricalCrossEntropy,MeanSquareError


def test_CategoricalCrossEntropy():
    print('start:CategoricalCrossEntropy')
    np.random.seed(1)
    Y_pred = np.random.randn(4,3)
    print(Y_pred)
    Y_label = np.array([0,1,0,1])
    print('CategoricalCrossEntropy.loss==>\n', CategoricalCrossEntropy.loss(Y_pred,Y_label))
    print('CategoricalCrossEntropy.diff==>\n', CategoricalCrossEntropy.diff(Y_pred, Y_label))
    print('end:CategoricalCrossEntropy')

def test_MeanSquareError():
    print('start:MeanSquareError')
    np.random.seed(1)
    Y_pred = np.random.randn(4, 1)
    print(Y_pred)
    Y_label = np.array([0, 1, 0, 1])
    print('MeanSquareError.loss==>\n', MeanSquareError.loss(Y_pred,Y_label))
    print('MeanSquareError.diff==>\n', MeanSquareError.diff(Y_pred, Y_label))
    print('end:MeanSquareError')


if __name__ == '__main__':
    test_CategoricalCrossEntropy()
    test_MeanSquareError()