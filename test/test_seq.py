import numpy as np

from src.container import Sequential
from src.layers import *


def test_Dense():
    s = Sequential()
    s.add(Dense(3,2))
    s.add(Activation('ReLU'))
    s.add(Dense(1))
    # s.add(Activation('sigmoid'))
    s.compile(objective='mse')
    np.random.randn(10)
    x_train = np.random.randn(10,2)
    y_train = np.zeros((10,1))
    y_train[(x_train[:, 0] < 0.5) & (x_train[:, 1] < 0.5)] = 1

    s.fit(x_train,y_train,epochs=10000)
    print(y_train)
    print(s.predict(x_train))

def test_Softmax():
    s = Sequential()
    s.add(Dense(3,2))
    s.add(Activation('ReLU'))
    s.add(Dense(2))
    s.add(Activation('softmax'))
    s.compile(objective='CategoricalCrossEntropy')
    np.random.randn(10)
    x_train = np.random.randn(10,2)
    y_train = np.zeros((10,1))
    y_train.dtype='int64'
    y_train[(x_train[:, 0] < 0.5) & (x_train[:, 1] < 0.5)] = 1
    s.fit(x_train,y_train, epochs=1000, learning_rate=0.01)
    print(y_train)
    print(s.predict(x_train))

if __name__ == '__main__':
    test_Dense()
    test_Softmax()
