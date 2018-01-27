import  numpy as np
from src.util import *
from src.layers.layer import Layer
import numba as nb
import time
class Pool(Layer):
    def __init__(self,  kernel_size, input_size=0, stride=1, mode='max', name=''):
        self.stride = stride
        self.kernel_size = kernel_size
        if mode not in ('max','average'):
            raise Exception('wrong pool mode, just support max and average')
        self.mode = mode

        # n_H_prev, n_W_prev, n_C_prev = input_size
        # f, f = kernel_size
        # n_H = int(1 + (n_H_prev - f) / self.stride)
        # n_W = int(1 + (n_W_prev - f) / self.stride)
        # super().__init__((n_H, n_W, n_C_prev), input_size, name)
        super().__init__(0, input_size, name)
    def build(self, loc_idx, input_size, init_param_method):
        super().build(loc_idx, input_size, init_param_method)
        n_H_prev ,n_W_prev, n_C_prev = input_size
        f, f = self.kernel_size
        n_H = int(1 + (n_H_prev - f) / self.stride)
        n_W = int(1 + (n_W_prev - f) / self.stride)
        self.output_size = self.node_size = (n_H, n_W, n_C_prev)

    @nb.jit
    def forward(self, X):
        # return speed_forward(X,self.kernel_size,self.stride,self.mode)

        m, n_H_prev, n_W_prev, n_C_prev = X.shape
        f, f = self.kernel_size
        n_H = int(1 + (n_H_prev - f) / self.stride)
        n_W = int(1 + (n_W_prev - f) / self.stride)
        n_C = n_C_prev
        # st = time.time()
        XX = img2col_HW(X, self.stride, f)
        # print('img2col_HW,cost', (time.time()-st))
        self.XX = XX
        if self.mode == 'max':
            Z = np.max(XX, axis=1)
        elif self.mode == 'average':
            Z = np.mean(XX, axis=1)
        return Z.reshape(m, n_H, n_W, n_C)

    @nb.jit
    def backward(self, X, dA):
        # return simple_backward(X,dA,self.input_size,self.kernel_size,self.stride,self.mode)
        m, n_H_prev, n_W_prev, n_C_prev = X.shape
        m, n_H, n_W, n_C = dA.shape
        dims = (m,) + self.input_size
        f, f = self.kernel_size
        dA = dA.reshape(m*n_H*n_W*n_C,1)
        dAA = np.zeros((m * n_H * n_W * n_C, f * f))

        if self.mode == 'max':
            XX_idx = np.argmax(self.XX, axis=1)
            dAA[range(dAA.shape[0]),XX_idx] = dA[range(dAA.shape[0]),0]
        elif self.mode == 'average':
            dAA = np.repeat(np.divide(dA, f*f), f*f, axis=1)

        return col2img_HW(dAA, (n_H_prev, n_W_prev, n_C_prev), self.stride, f), None, None

def simple_forward(X, kernel_size, stride, mode):
    m, n_H_prev, n_W_prev, n_C_prev = X.shape
    f, f = kernel_size
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    A = np.zeros((m, n_H, n_W, n_C))
    for i in range(m):
        # loop window square
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # select window location
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_prev_slice = X[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    return A

def simple_backward(X, dA, input_size, kernel_size,stride, mode):
    m, n_H, n_W, n_C = dA.shape
    dims = (m,) + input_size
    dA_prev = np.zeros(dims)
    f, f = kernel_size
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # to mask window of max
                    if mode == 'max':
                        mask = X[i,vert_start:vert_end,horiz_start:horiz_end,c] == np.max(X[i,vert_start:vert_end,horiz_start:horiz_end,c])
                        a_prev_slice = X[i, vert_start:vert_end, horiz_start:horiz_end, c]
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i,h,w,c])
                    elif mode == 'average':
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.divide(dA[i,h,w,c], f*f)
    return dA_prev, None, None

if __name__ == '__main__':


    # input_size = (4,4,3)
    # np.random.seed(1)
    # A_prev = np.random.randn(2, *(input_size) )
    # pool = Pool((4,4),input_size)
    # print(pool.forward(A_prev))
    # pool.mode='average'
    # print(pool.forward(A_prev))
    # print('=====================')

    input_size = (5, 3, 2)
    np.random.seed(1)
    A_prev = np.random.randn(5, 5, 3, 2)
    dA = np.random.randn(5, 4, 2, 2)
    print(dA[0,0])
    pool = Pool((2,2))
    pool.build(0,(5, 3, 2),'random')
    A = pool.forward(A_prev)
    print('A[0,0]=',A[0,0])
    dA_prev,_,_ = pool.backward(A_prev, dA)
    print("mode = max")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1, 1])

    pool.mode = 'average'
    dA_prev,_,_ = pool.backward(A_prev, dA)
    print("mode = average")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ',dA_prev[1,1])

