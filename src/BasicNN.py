import numpy as np


def default_func(A):
    return np.ones(A.shape)
class BasicNN(object):
    """ Basic Neural Network

    """
    def __init__(self,node_size,input_size=0,init_parameter='random'):
        self.node_size = node_size
        self.input_size = input_size
        self.W = np.random.rand(input_size, node_size) * 0.01
        self.b = np.zeros((1,node_size))
    def set_input_size(self,input_size):
        self.W = np.random.rand(input_size, self.node_size) * 0.01
        self.input_size = input_size
    def forward(self, input_data):

        self.pre_data = input_data
        self.data_forward = np.add(np.dot(input_data,self.W), self.b)
        return self.data_forward

    def backward(self, dZ, func=default_func):
        m = dZ.shape[0]
        grad_W = np.dot(self.pre_data.T, dZ) / m
        grad_b = np.sum(dZ, axis=0, keepdims=True) / m
        # print('db_shape', self.b,np.sum(dZ, axis=0) / m)
        dZ = np.dot(dZ, self.W.T)
        assert grad_W.shape == (self.input_size, self.node_size)
        assert grad_b.shape == (1, self.node_size)

        return grad_W, grad_b, dZ

    # def backward(self, dZ, pre_W, func=default_func):
    #     """ dZ.shape=sample,node_size
    #     pre_W.shape = node_size,back_layer_size
    #
    #     :param dZ:
    #     :param pre_W:
    #     :return:
    #     """
    #     m = dZ.shape[0]
    #     input_size, node_size = self.W.shape
    #     _, back_size = pre_W.shape
    #     grad_b = np.zeros((m, node_size))
    #     grad_W = np.zeros((m, input_size, node_size))
    #     dX = np.zeros((m, input_size))
    #     for i in range(m):
    #         for j in range(node_size):
    #             for k in range(back_size):
    #                 tmp = dZ[i,j] * pre_W[j,k] * func(self.data_forward[i,j])
    #                 grad_b[i,j] += tmp
    #                 grad_W[i,:,j] += tmp * self.pre_data[i]
    #                 # print('tmp:',tmp,self.W[:,j])
    #                 # print('dX:',dX[i])
    #                 dX[i] += self.W[:,j] * tmp
    #     assert grad_W.shape[1:] == (self.input_size,self.node_size)
    #     assert grad_b.shape[1] == self.node_size
    #     return grad_W, grad_b, dX

if __name__ == '__main__':
    nn = BasicNN(3,4)
    np.random.seed(10)
    input_data = np.random.rand(2,4)
    print('input:',input_data)
    print('param:',nn.W,nn.b)
    print('forward=',nn.forward(input_data))

    dZ = np.random.rand(2,3)
    pre_W = np.random.rand(3,1)
    print('dZ:',dZ)
    print('pre_W:',pre_W)
    back =nn.backward(dZ)

    print('backward_W:',back[0])
    print('backward_b:',back[1])
    print('backward_X:',back[2])