# dot_cython.pyx
import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.double_t, ndim=4] _speed_forward(np.ndarray[np.double_t, ndim=4] X, np.ndarray[np.double_t, ndim=4] W,
                                        np.ndarray[np.double_t, ndim=4] b,tuple output_size, int stride, int pad):
    cdef np.ndarray[np.double_t, ndim=4] Z, X_pad
    cdef np.ndarray[np.double_t, ndim=3] A_slice_prev
    cdef int m, n_H_prev, n_W_prev, n_C_prev
    cdef f, n_H, n_W, n_C

    m, n_H_prev, n_W_prev, n_C_prev = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
    f, f, n_C_prev, n_C = W.shape[0], W.shape[1], W.shape[2], W.shape[3]
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    n_H, n_W, n_C = output_size

    Z = np.zeros((m, n_H, n_W, n_C))
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    A_slice_prev = X_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]
                    Z[i, h, w, c] = np.sum(np.multiply(A_slice_prev, W[..., c]) + b[..., c])
                    # Z[i, h, w, c] = conv_single_step(A_slice_prev, W[..., c], b[..., c])
    return Z

def speed_forward(X, W, b,output_size, stride, pad):
    return _speed_forward(X, W, b,output_size, stride, pad)