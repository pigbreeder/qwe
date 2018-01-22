from src.layers.layer import Layer
from src.unit import BasicUnit
import src.initialization
import numpy as np

class Convolution2D(Layer):
    # input_size:tuple or list include height,width,channel
    def __init__(self, filters, kernel_size, input_size=0, name='', pad=0, stride=1):
        super().__init__(kernel_size +(filters, ), input_size, name)
        self.pad = pad
        self.filters = filters
        self.stride = stride
        self.kernel_size = kernel_size

    def build(self, loc_idx, input_size, init_param_method):
        pass
        super().build(loc_idx, input_size, init_param_method)

        self.W = self.kernel_size + (self.input_size[2], self.filters)
        self.b = np.zeros((1, 1, self.input_size[2], self.filters))
        self.W = src.initialization.get(init_param_method)((*self.kernel_size,self.input_size[2], self.filters))

        n_H_prev, n_W_prev, n_C_prev = self.input_size
        (f, f, n_C_prev, n_C) = self.W.shape
        n_H = int((n_H_prev - f + 2 * self.pad) / self.stride) + 1
        n_W = int((n_W_prev - f + 2 * self.pad) / self.stride) + 1
        self.output_size = (n_H, n_W, n_C)
        # self.b = np.zeros(1, self.output_size)

    def forward(self, X):
        m, n_H_prev, n_W_prev, n_C_prev = X.shape
        (f, f, n_C_prev, n_C) = self.W.shape
        n_H = int((n_H_prev - f + 2 * self.pad) / self.stride) + 1
        n_W = int((n_W_prev - f + 2 * self.pad) / self.stride) + 1
        n_H, n_W, n_C = self.output_size

        Z = np.zeros((m, n_H, n_W, n_C))
        X_pad = self.zero_pad(X, self.pad)
        for i in range(m):
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        vert_start = h * self.stride
                        vert_end = vert_start + f
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + f
                        A_slice_prev = X_pad[i,vert_start:vert_end, horiz_start:horiz_end, :]
                        Z[i,h,w,c] = self.conv_single_step(A_slice_prev, self.W[...,c], self.b[...,c])

        return Z

    def backward(self, X, dA):
        pass
        m, n_H, n_W, n_C = dA.shape
        f, f, n_C_prev, n_C = self.W.shape

        dA_prev = np.zeros(X.shape)
        dW = np.zeros(self.W.shape)
        db = np.zeros(self.b.shape)

        A_prev_pad = self.zero_pad(X, self.pad)
        dA_prev_pad = self.zero_pad(dA_prev, self.pad)

        for i in range(m):
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        vert_start = h
                        vert_end = vert_start + f
                        horiz_start = w
                        horiz_end = horiz_start + f
                        a_slice = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]

                        dA_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :] += self.W[...,c] * dA[i,h,w,c]
                        dW[...,c] += a_slice * dA[i,h,w,c]
                        db[...,c] += dA[i,h,w,c]
            if self.pad > 0:
                dA_prev[i,...] = dA_prev_pad[i, self.pad:-self.pad, self.pad:-self.pad, :]
            else:
                dA_prev[i,...] = dA_prev_pad[i,...]
        return dA_prev, dW, db


    def conv_single_step(self, X, W, b):
        # 对一个裁剪图像进行卷积
        # X.shape = f, f, prev_channel_size
        return np.sum(np.multiply(X, W) + b)
    def zero_pad(self, X, pad):
        """
        X -- shape (m, n_H, n_W, n_C)
        Returns:
        X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
        """
        X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
        return X_pad

if __name__ == '__main__':
    pass
    con = Convolution2D(8,(2,2), pad=2)
    con.build(1,(4,4,3),'randn')
    np.random.seed(1)
    A_prev = np.random.randn(10, 4, 4, 3)
    W = np.random.randn(2, 2, 3, 8)
    b = np.random.randn(1, 1, 1, 8)
    con.W = W
    con.b = b
    ret = con.forward(A_prev)
    print(ret.shape)
    print(np.mean(ret))
    np.random.seed(1)
    dA, dW, db = con.backward(A_prev, ret)
    # print(ret)
    print("dA_mean =", np.mean(dA),dA.shape)
    print("dW_mean =", np.mean(dW),dW.shape)
    print("db_mean =", np.mean(db),db.shape)
    print(dW)
