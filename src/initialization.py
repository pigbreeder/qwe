import numpy as np

def randn(shape, scale=0.01):
    return np.random.randn(*shape) * scale

def zeros(shape, dtype=np.double):
    return np.zeros(shape, dtype)


def uniform(shape, scale=0.05):
    return np.random.uniform(low=-scale, high=scale, size=shape)


def normal(shape, scale=0.05):
    return np.random.normal(loc=0.0, scale=scale, size=shape)


def glorot_normal(shape):
    if len(shape) == 2:
        nb_in, nb_out = shape[0], shape[1]
    # 4维KN*KC*KH*KW数据格式
    elif len(shape) == 4:
        nb_in, nb_out = np.product(shape[1:]), shape[0]
    else:
        raise Exception("Invalid shape: shape's length must be 2D!")

    s = np.sqrt(2.0 / (nb_in + nb_out))
    return np.random.normal(loc=0.0, scale=s, size=shape)


_dict = {'zeros': zeros,
         'randn': randn,
         'uniform': uniform,
         'normal': normal,
         'glorot_normal': glorot_normal,
         }

def get(name):
    val = _dict.get(name, None)
    if val == None:
        raise Exception(('the initialization method %s is not existed.' % name))
    return val
