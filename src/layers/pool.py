import  numpy as np

from src.layers.layer import Layer


class Pool(Layer):
    def __init__(self,  kernel_size, input_size, stride=1, mode='max', name=''):
        self.stride = stride
        self.kernel_size = kernel_size
        self.mode = mode
        n_H_prev ,n_W_prev, n_C_prev = input_size
        f = kernel_size
        n_H = int(1 + (n_H_prev - f) / self.stride)
        n_W = int(1 + (n_W_prev - f) / self.stride)
        super().__init__((n_H, n_W, n_C_prev), input_size, name)


    def forward(self, X):
        m, n_H_prev, n_W_prev, n_C_prev = X.shape
        f = self.kernel_size
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
        m, n_H, n_W, n_C = X.shape
        dims = (m,) + self.input_size
        dA_prev = np.zeros(dims)
        f = self.kernel_size
        for i in range(m):
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        vert_start = h
                        vert_end = vert_start + f
                        horiz_start = w
                        horiz_end = horiz_start + f

                        # to mask window of max
                        if self.mode == 'max':
                            mask = X[i,vert_start:vert_end,horiz_start:horiz_end,c] == np.max(X[i,vert_start:vert_end,horiz_start:horiz_end,c])
                            a_prev_slice = X[i, vert_start:vert_end, horiz_start:horiz_end, c]
                            dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i,h,w,c])
                        elif self.mode == 'average':
                            dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.divide(dA[i,h,w,c], f*f)
        return dA_prev

if __name__ == '__main__':

    # np.random.seed(1)
    # input_size = (4,4,3)
    # A_prev = np.random.rand(*((2,) + input_size) )
    # pool = Pool(4,input_size)
    # print(pool.forward(A_prev))
    # pool.mode='average'
    # print(pool.forward(A_prev))

    np.random.seed(1)
    input_size = (5, 3, 2)
    A_prev = np.random.randn(5, 5, 3, 2)
    pool = Pool(2, (5, 3, 2))
    pool.forward(A_prev)
    dA = np.random.randn(5,4,2,2)
    A_prev = pool.forward(dA)
    dA = np.random.randn(5, 4, 2, 2)

    dA_prev = pool.backward(A_prev, dA)
    print("mode = max")
    print('mean of dA = ', np.mean(dA))

    pool.mode = 'average'
    dA_prev = pool.backward(A_prev, dA)
    print("mode = average")
    print('mean of dA = ', np.mean(dA))
    print(dA_prev[1,1])
