import numpy as np

from src.BasicNN import BasicNN


class Activation(BasicNN):
    def __init__(self, method='ReLU'):
        if method == 'sigmoid':
            self.func = self.sigmoid
            self.der = self.sigmoid_derivative
        elif method == 'ReLU':
            self.func = self.ReLU
            self.der = self.ReLU_derivative

    def set_input_size(self,input_size):
        self.node_size = input_size
        self.input_size = input_size
        self.W = np.ones((input_size, input_size))
        self.b = np.zeros((1, input_size))

    def ReLU(self, z):
        A = np.maximum(0, z)
        return A


    def ReLU_derivative(self, z):
        dZ = np.array(z,copy=True)
        dZ[z <= 0] = 0
        return dZ

    def sigmoid(self, z):
        return 1.0/(1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def forward(self, input_data):
        z = super(Activation, self).forward(input_data)
        self.data_forward = self.func(z)
        return self.data_forward

    def backward(self, dZ, pre_W):
        return super(Activation, self).backward(dZ, pre_W, func=self.der)

if __name__ == '__main__':
    pass
    activation = Activation()
    activation.set_input_size(3)
    np.random.seed(10)
    input_data = np.random.rand(2, 3)
    print('input_data:',input_data)
    print('forward=', activation.forward(input_data))
    dZ = np.random.rand(2, 3)
    pre_W = np.random.rand(3, 1)
    back = activation.backward(dZ, pre_W)

    print('backward_W:', back[0])
    print('backward_b:', back[1])