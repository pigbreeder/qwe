import numpy as np




def img2col(X, pad, stride, f):
    pass
    ff = f * f
    m, n_H_prev, n_W_prev, n_C_prev= X.shape
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
                    t = col // n_C_prev
                    hh = t // f
                    ww = t % f
                    cc = col % n_C_prev
                    Z[row, col] = X_pad[i, vert_start + hh, horiz_start + ww, cc]
    return Z


def col2img(X, output_size, pad, stride, f):
    pass
    ff = f * f
    n_H_prev, n_W_prev, n_C_prev = output_size
    n_X_H_prev, n_X_W_prev = X.shape
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
                    t = col // n_C_prev
                    hh = t // f
                    ww = t % f
                    cc = col % n_C_prev
                    Z[i, vert_start + hh, horiz_start + ww, cc] += X[row, col]
    if pad > 0:
        return Z[:, pad:-pad, pad:-pad, :]
    else:
        return Z

def img2col_HW(X,stride, f):
    pass
    m, n_H_prev, n_W_prev, n_C_prev= X.shape
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


def col2img_HW(X, output_size, stride, f):
    pass
    n_H_prev, n_W_prev, n_C_prev = output_size
    n_X_H_prev, n_X_W_prev = X.shape
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