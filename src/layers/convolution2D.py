# encoding: utf-8
from src.layers.layer import Layer
from src.unit import BasicUnit
import src.initialization
import numpy as np
import numba as nb
from src.util import *
import time

class Convolution2D(Layer):
    # input_size:tuple or list include height,width,channel
    def __init__(self, filters, kernel_size, input_size=0, name='', pad=0, stride=1):
        super().__init__(kernel_size +(filters, ), input_size, name)
        self.pad = pad
        self.filters = filters
        self.stride = stride
        self.kernel_size = kernel_size

    def build(self, loc_idx, input_size, init_param_method='glorot_normal'):
        pass
        super().build(loc_idx, input_size, init_param_method)

        # self.W = self.kernel_size + (self.input_size[2], self.filters)
        self.b = np.zeros((1, 1, 1, self.filters))
        self.W = src.initialization.get(init_param_method)((self.input_size[2], *self.kernel_size, self.filters))
        # self.W = src.initialization.get(init_param_method)(self.kernel_size + (self.input_size[2], self.filters))

        n_H_prev, n_W_prev ,n_C_prev = self.input_size
        (n_C_prev, f, f, n_C) = self.W.shape
        n_H = int((n_H_prev - f + 2 * self.pad) / self.stride) + 1
        n_W = int((n_W_prev - f + 2 * self.pad) / self.stride) + 1
        self.output_size = (n_H, n_W, n_C)

    @nb.jit
    def forward(self, X):
        W = self.W
        b = self.b
        stride = self.stride
        pad = self.pad
        (n_C_prev, f, f, n_C) = W.shape
        m, n_H_prev, n_W_prev, n_C_prev = X.shape

        n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
        n_W = int((n_W_prev - f + 2 * pad) / stride) + 1


        # X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
        # Z = np.zeros((m * n_H * n_W, f * f * n_C_prev), dtype=np.double)
        # st = time.time()
        # XX = src.ext.sample_pyx.img2col(X_pad, Z,n_H,n_W, stride, f)
        # XX = img2col(X_pad, Z, n_H, n_W, stride, f)

        # 先将X转换为二维矩阵，然后进行前向乘积运算，加快速度。原始实现方法看simple_convolution2D.py
        XX = img2col(X, pad, stride, f)
        # print('cnn ,forward,',(time.time()-st))
        # WW = WW.reshape(f*f*n_C_prev, n_C)
        WW = W.reshape(f * f * n_C_prev, n_C)
        self.XX = XX
        self.WW = WW

        Z = BasicUnit.forward(XX, WW, b)
        return Z.reshape(m, n_H, n_W, n_C)


    def backward(self, X, dA):
        pass
        return speed_backward(self, X,dA)



def speed_forward(model, X):
    W = model.W
    b = model.b
    stride = model.stride
    pad = model.pad
    (n_C_prev, f, f, n_C) = W.shape
    m, n_H_prev, n_W_prev, n_C_prev = X.shape

    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

    # WW = W.swapaxes(2,1)
    # WW = WW.swapaxes(1,0)

    XX = img2col(X, pad, stride, f)
    # WW = WW.reshape(f*f*n_C_prev, n_C)
    WW = W.reshape(f*f*n_C_prev, n_C)
    model.XX = XX
    model.WW = WW

    Z = BasicUnit.forward(XX, WW, b)
    return Z.reshape(m, n_H, n_W, n_C)
@nb.jit
def speed_backward(model,X, dA):
    W = model.W
    b = model.b
    stride = model.stride
    pad = model.pad

    (n_C_prev, f, f, n_C) = W.shape
    m, n_H_prev, n_W_prev, n_C_prev = X.shape

    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

    dA = dA.reshape(m * n_H * n_W, n_C)
    dAA, dW, db = BasicUnit.backward(model.XX, dA, model.WW)
    # Z = np.zeros((m, n_H_prev + 2 * pad, n_W_prev + 2 * pad, n_C_prev), dtype=np.float64)

    # 将dAA回原先的m,h,w,c格式。原始实现方法看simple_convolution2D.py
    dAA = col2img(dAA, (n_H_prev, n_W_prev, n_C_prev), pad, stride, f)
    # dAA = col2img(dAA,Z, (n_H_prev, n_W_prev, n_C_prev), n_H, n_W, pad, stride, f)
    dW = dW.reshape(n_C_prev, f, f, n_C)

    return dAA, dW, db

@nb.jit(nopython=True)
def conv_single_step(X, W, b):
    # 对一个裁剪图像进行卷积
    # X.shape = f, f, prev_channel_size
    return np.sum(np.multiply(X, W) + b)


# @nb.jit('double[:,:,:](double[:,:,:],int16)', nopython=True)
def zero_pad(X, pad):
    """
    X -- shape (m, n_H, n_W, n_C)
    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
    return X_pad

if __name__ == '__main__':
    pass
    import time
    st = time.time()
    for i in range(1000):
        con = Convolution2D(8,(2,2), pad=2)
        con.build(1,(4,4,3),'randn')
        np.random.seed(1)
        A_prev = np.random.randn(10, 4, 4, 3)
        W = np.random.randn(3, 2, 2, 8)
        b = np.random.randn(1, 1, 1, 8)
        con.W = W
        con.b = b
        ret = con.forward(A_prev)
        # print(A_prev[0])
        # print(ret[0])
        print(ret.shape,i)
        print(np.mean(ret))
        np.random.seed(1)
        dA, dW, db = con.backward(A_prev, ret)
        # print("dA_mean =", np.mean(dA),dA.shape)
        # print("dW_mean =", np.mean(dW),dW.shape)
        # print("db_mean =", np.mean(db),db.shape)
    print(time.time()-st)