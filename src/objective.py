import numpy as np
"""
计算损失函数要得到均值
计算损失函数导数不要取均值
"""
class Objective(object):
    @staticmethod
    def loss(Y_pred, Y_label):
        pass

    @staticmethod
    def diff(Y_pred, Y_label):
        pass

class MeanSquareError(Objective):
    """
    function = 1/2m * (pred-label)^2
    derivative = 1/m * (pred - label)
    """

    # shape = (sample_size,)

    @staticmethod
    def loss(Y_pred, Y_label):
        return np.mean(np.square(Y_pred - Y_label))

    @staticmethod
    def diff(Y_pred, Y_label):
        # return ans.shape = (1, category_size)
        return (Y_pred - Y_label)

class CategoricalCrossEntropy(Objective):

    # Y_pred.shape = (sample_size, category_size)
    # Y_label.shape = (sample_size,)

    @staticmethod
    def predict(X):
        Y_pred_exp = np.exp(X)
        return np.divide(Y_pred_exp, np.sum(Y_pred_exp, axis=1, keepdims=True))

    @staticmethod
    def loss(Y_pred, Y_label):
        Y_pred_source = CategoricalCrossEntropy.predict(Y_pred)
        return np.mean(-np.log(Y_pred_source[range(Y_pred.shape[0]), Y_label]))


    @staticmethod
    def diff(Y_pred, Y_label):
        # return ans.shape = (1, category_size)
        Y_pred_source = CategoricalCrossEntropy.predict(Y_pred)
        Y_pred_source[range(Y_pred.shape[0]), Y_label] -= 1
        # return np.mean(Y_pred_source, axis=0)
        return Y_pred_source
class LogisticCrossEntropy(Objective):
    @staticmethod
    def predict(X):
        return X

    @staticmethod
    def loss(Y_pred, Y_label):
        super().loss(Y_pred, Y_label)

    @staticmethod
    def diff(Y_pred, Y_label):
        super().diff(Y_pred, Y_label)

_dict = {'MeanSquareError': MeanSquareError,
         'mse': MeanSquareError,
         'CategoricalCrossEntropy': CategoricalCrossEntropy,
         }


def get(name):
    val = _dict.get(name, None)
    if val == None:
        raise Exception(('the loss function %s is not existed.' % name))
    return val