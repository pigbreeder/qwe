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

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.double_t, ndim=2]_img2col(np.ndarray[np.double_t, ndim=4] X, int pad, int stride, int f):
    cdef np.ndarray[np.double_t, ndim = 2] Z
    cdef np.ndarray[np.double_t, ndim = 4] X_pad
    cdef int ff, m, n_H_prev, n_W_prev, n_C_prev,i,h,w,c,row,vert_start,horiz_start,t,hh,ww,cc

    ff = f * f
    m, n_H_prev, n_W_prev, n_C_prev= X.shape[0],X.shape[1],X.shape[2],X.shape[3]
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    Z = np.zeros((m * n_H * n_W, f * f * n_C_prev))
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
    row = -1

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                row += 1
                vert_start = h * stride
                horiz_start = w * stride
                for col in range(f * f * n_C_prev):
                    t = col % ff
                    hh = t // f
                    ww = t % f
                    cc = col // ff
                    Z[row, col] = X_pad[i, vert_start + hh, horiz_start + ww, cc]
    return Z



@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.double_t, ndim=4] _col2img(np.ndarray[np.double_t, ndim=2] X, tuple output_size, int pad, int stride, int f):

    cdef np.ndarray[np.double_t, ndim = 4] Z
    cdef int ff, m, n_H_prev, n_W_prev, n_C_prev,i,h,w,c,row,vert_start,horiz_start,t,hh,ww,cc

    ff = f * f
    n_H_prev, n_W_prev, n_C_prev = output_size[0],output_size[1],output_size[2]
    n_X_H_prev, n_X_W_prev = X.shape[0],X.shape[1]
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

    n_C = n_X_W_prev // ff
    m = n_X_H_prev // (n_H * n_W)
    Z = np.zeros((m, n_H_prev + 2*pad, n_W_prev + 2*pad, n_C_prev))
    row = -1
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                row += 1
                vert_start = h * stride
                horiz_start = w * stride
                for col in range(f * f * n_C_prev):
                    t = col % ff
                    hh = t // f
                    ww = t % f
                    cc = col // ff
                    Z[i, vert_start + hh, horiz_start + ww, cc] += X[row, col]
    if pad > 0:
        return Z[:, pad:-pad, pad:-pad, :]
    else:
        return Z



@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.double_t, ndim=2] _img2col_HW(np.ndarray[np.double_t, ndim=4] X,int stride, int f):
    cdef np.ndarray[np.double_t, ndim=2] Z
    cdef int m,n_H_prev,n_W_prev,n_C_prev,n_H,n_W
    cdef i, h, w, c, horiz_start, vert_start, row, col, hh, ww
    m, n_H_prev, n_W_prev, n_C_prev= X.shape[0],X.shape[1],X.shape[2],X.shape[3]
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)

    Z = np.zeros((m * n_H * n_W * n_C_prev, f * f))
    row = -1

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                vert_start = h * stride
                horiz_start = w * stride
                for c in range(n_C_prev):
                    row += 1
                    for col in range(f * f):
                        hh = col // f
                        ww = col % f
                        Z[row, col] = X[i, vert_start + hh, horiz_start + ww, c]
    return Z

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.double_t, ndim = 4] _col2img_HW(np.ndarray[np.double_t, ndim = 2] X, tuple output_size, int stride, int f):

    cdef np.ndarray[np.double_t, ndim = 4] Z
    cdef int m, n_H_prev, n_W_prev, n_C_prev, n_H, n_W
    cdef i, h, w, c, horiz_start, vert_start, row, col, hh, ww

    n_H_prev, n_W_prev, n_C_prev = output_size[0],output_size[1],output_size[2]
    n_X_H_prev, n_X_W_prev = X.shape[0],X.shape[1]
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)

    m = n_X_H_prev // (n_H * n_W * n_C_prev)
    Z = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    row = -1
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                vert_start = h * stride
                horiz_start = w * stride
                for c in range(n_C_prev):
                    row += 1
                    for col in range(f * f):
                        hh = col // f
                        ww = col % f
                        Z[i, vert_start + hh, horiz_start + ww, c] += X[row, col]
    return Z

def img2col(X, pad, stride, f):
    return _img2col(X,pad,stride,f)

def col2img(X, output_size, pad, stride, f):
    return _col2img(X, output_size, pad, stride, f)

def img2col_HW(X,stride, f):
    return _img2col_HW(X,stride,f)

def col2img_HW(X, output_size, stride, f):
    return _col2img_HW(X,output_size,stride,f)