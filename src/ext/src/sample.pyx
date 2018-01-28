

# dot_cython.pyx
import numpy as np
cimport numpy as np
cimport cython



@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.double_t, ndim=2] __img2col(np.ndarray[np.double_t, ndim=4] X_pad, np.ndarray[np.double_t, ndim=2] Z, int n_H, int n_W, int stride, int f):

    cdef int m, n_C_prev,i,h,w,c,row,vert_start,horiz_start,hh,ww,cc
    cdef double t

    m, _, _, n_C_prev = X_pad.shape[0],X_pad.shape[1],X_pad.shape[2],X_pad.shape[3]
    # m = 64
    # n_C_prev=1
    row = -1
    cdef double *ZZ = &Z[0,0]
    cdef double *XX = &X_pad[0,0,0,0]

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                row += 1
                col=0
                vert_start = h * stride
                horiz_start = w * stride

                for hh in range(f):
                    for ww in range(f):
                        for cc in range(n_C_prev):
                            # dim 1
                            # Z[row, col] = X_pad[i, vert_start + hh, horiz_start + ww, cc]
                            ZZ[row * f * f * n_C_prev + col] = \
                                XX[i * n_H * n_W * n_C_prev + (vert_start + hh) * n_W * n_C_prev + (horiz_start + ww) * n_C_prev + cc]
                            col += 1
    return Z

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.double_t, ndim=2]  _o_img2col(np.ndarray[np.double_t, ndim=4] X, int pad, int stride, int f):
    cdef np.ndarray[np.double_t, ndim = 2] Z
    cdef np.ndarray[np.double_t, ndim = 4] X_pad
    cdef int ff, m, n_H_prev, n_W_prev, n_C_prev,i,h,w,c,row,vert_start,horiz_start,t,hh,ww,cc

    ff = f * f
    m, n_H_prev, n_W_prev, n_C_prev= X.shape[0],X.shape[1],X.shape[2],X.shape[3]
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    Z = np.zeros((m * n_H * n_W, f * f * n_C_prev), dtype=np.double)
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
    cdef double * XX = &X_pad[0,0,0, 0]
    cdef double * ZZ = &Z[0, 0]
    row = -1

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                row += 1
                vert_start = h * stride
                horiz_start = w * stride
                col=0
                for hh in range(f):
                    for ww in range(f):
                        for cc in range(n_C_prev):
                            # Z[row, col] = X_pad[i, vert_start + hh, horiz_start + ww, cc]
                            ZZ[row * f * f * n_C_prev + col] = \
                                XX[i * n_H * n_W * n_C_prev + (vert_start + hh) * n_W * n_C_prev + (
                                        horiz_start + ww) * n_C_prev + cc]
                            col += 1
    return Z

def img2col(X_pad,Z, n_H, n_W, stride, f):
    return __img2col(X_pad,Z, n_H, n_W,stride,f)
def o_img2col(X, pad, stride, f):
    return _o_img2col(X,pad,stride,f)