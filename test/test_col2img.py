from src.util import img2col, col2img
import numpy as np

def test_img2col():
    np.random.seed(1)
    X = np.random.randn(2,3,3,1)
    pad = 1
    stride = 1
    f = 2
    r = img2col(X,pad,stride,f)
    print(X)
    print('===')
    print(r)
    print('===')
    W = np.arange(8)
    XX = np.dot(r,W.reshape(4,2))
    print(XX)
    print(XX.reshape(2,4,4,2))
    return r

def test_col2img(r):
    pad = 1
    stride = 1
    f = 2
    r = col2img(r,(3,3,1),pad,stride,f)
    print('===')
    print(r)

r= test_img2col()
# test_col2img(r)