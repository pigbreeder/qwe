import numpy as np
from config.basic import *
if SWITCH_EXT:
    from src.ext import *
else:
    from .py_util import *

def grad_clicp(grad,th_max=1e6):
    # print('in grad_clicp', grad)
    # t = np.linalg.norm(grad)
    # if t > th_max:
    #     print('grad max')
    #     return np.divide(th_max, t) * grad
        # return np.divide(th_min, t) * grad
    return grad
    # return np.clip(grad, th_min, th_max)

import matplotlib.pyplot as plt

# Helper function to plot a decision boundary.
def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    print(Z.shape)
    Z = Z.reshape(xx.shape)
    print(Z.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
