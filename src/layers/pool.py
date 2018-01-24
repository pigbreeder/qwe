import  numpy as np

from src.layers.layer import Layer
import numba as nb

class Pool(Layer):
    def __init__(self,  kernel_size, input_size=0, stride=1, mode='max', name=''):
        self.stride = stride
        self.kernel_size = kernel_size
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


    def forward(self, X):
        return speed_forward(X,self.kernel_size,self.stride,self.mode)

        m, n_H_prev, n_W_prev, n_C_prev = X.shape
        f, f = self.kernel_size
        n_H = int(1 + (n_H_prev - f) / self.stride)
        n_W = int(1 + (n_W_prev - f) / self.stride)
        n_C = n_C_prev
        A = np.zeros((m, n_H, n_W, n_C))
        for i in range(m):
            # loop window square
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        # select window location
                        vert_start = h * self.stride
                        vert_end = vert_start + f
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + f

                        a_prev_slice = X[i, vert_start:vert_end, horiz_start:horiz_end, c]

                        if self.mode == "max":
                            A[i, h, w, c] = np.max(a_prev_slice)
                        elif self.mode == "average":
                            A[i, h, w, c] = np.mean(a_prev_slice)
        return A

    def backward(self, X, dA):
        return speed_backward(X,dA,self.input_size,self.kernel_size,self.stride,self.mode)

        m, n_H, n_W, n_C = dA.shape
        dims = (m,) + self.input_size
        dA_prev = np.zeros(dims)
        f, f = self.kernel_size
        for i in range(m):
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        vert_start = h * self.stride
                        vert_end = vert_start + f
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + f

                        # to mask window of max
                        if self.mode == 'max':
                            mask = X[i,vert_start:vert_end,horiz_start:horiz_end,c] == np.max(X[i,vert_start:vert_end,horiz_start:horiz_end,c])
                            a_prev_slice = X[i, vert_start:vert_end, horiz_start:horiz_end, c]
                            dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i,h,w,c])
                        elif self.mode == 'average':
                            dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.divide(dA[i,h,w,c], f*f)
        return dA_prev, None, None
@nb.jit
def speed_forward(X, kernel_size, stride, mode):
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
@nb.jit
def speed_backward(X, dA, input_size, kernel_size,stride, mode):
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
@nb.jit(nopython=True)
def numba_str(txt, sample):
    x=0
    for i in range(txt.size):
        if txt[i]==sample:
            x += 1
    return x
if __name__ == '__main__':


    # input_size = (4,4,3)
    # np.random.seed(1)
    # A_prev = np.random.randn(2, *(input_size) )
    # pool = Pool((4,4),input_size)
    # print(pool.forward(A_prev))
    # pool.mode='average'
    # print(pool.forward(A_prev))


    input_size = (5, 3, 2)
    np.random.seed(1)
    A_prev = np.random.randn(5, 5, 3, 2)
    dA = np.random.randn(5, 4, 2, 2)
    print(dA[0,0])
    pool = Pool((2,2))
    pool.build(0,(5, 3, 2),'random')
    A = pool.forward(A_prev)
    # print(A)
    dA_prev,_,_ = pool.backward(A_prev, dA)
    print("mode = max")
    print('mean of dA = ', np.mean(dA_prev))


    pool.mode = 'average'
    dA_prev,_,_ = pool.backward(A_prev, dA)
    print("mode = average")
    print('mean of dA = ', np.mean(dA_prev))
    print(dA_prev[1,1])
