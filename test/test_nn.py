import numpy as np
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt
from src.container import Sequential
from src.layers import *
from src.util import plot_decision_boundary


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

    s.fit(x_train,y_train,epochs=10000,learning_rate=0.09)
    print(y_train)
    print(s.predict(x_train))

def test_Softmax():
    s = Sequential()
    s.add(Dense(10,input_size=2))
    s.add(Activation('tanh'))
    s.add(Dense(5))
    s.add(Activation('tanh'))
    s.add(Dense(2))
    s.add(Activation('tanh'))
    s.add(Activation('softmax'))
    s.compile(optimizer='bgd_reg', objective='CategoricalCrossEntropy')
    np.random.seed(0)
    x_train, y_train = sklearn.datasets.make_moons(200, noise=0.20)
    # plt.scatter(x_train[:,0], x_train[:,1], s=40, c=y_train, cmap=plt.cm.Spectral)
    # plt.show()
    s.fit(x_train,y_train, epochs=10000, learning_rate=0.01,batch_size=100)
    plot_decision_boundary(lambda x: np.argmax(s.predict(x), axis=1), x_train, y_train)
    plt.title("Decision Boundary for hidden layer size 3")
    plt.show()
    # print(s.predict(x_train))

if __name__ == '__main__':
    # test_Dense()
    test_Softmax()
