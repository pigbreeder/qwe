from abc import ABCMeta, abstractmethod

import numpy as np


class Optimizer(object):
    __metaclass__ = ABCMeta

    def __init__(self, model):
        pass
        self.model = model

    def set_param(self, n_epoch=1000, learning_rate=0.01, batch_size=64, switch_print=True, iter_times_print=100):
        self.learning_rate = learning_rate
        self.iterations = 0
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.switch_print = switch_print
        self.iter_times_print = iter_times_print

    @abstractmethod
    def update(self, params, grads):
        pass

    @abstractmethod
    def iterate(self, x_train, y_train):
        pass


class BGD(Optimizer):

    def update(self, params, grads):
        super().update(params, grads)
        for p, d in zip(params, grads):
            pw, pb = p
            dw, db = d

            pw -= self.learning_rate * dw
            pb -= self.learning_rate * db

    def iterate(self, x_train, y_train):
        pass
        sample_size = x_train.shape[0]
        for epoch in range(1, self.n_epoch + 1):
            for i in range(0, sample_size, self.batch_size):
                j = i + self.batch_size
                if j > sample_size:
                    j = sample_size
                x_train_data = x_train[i:j]
                y_train_data = y_train[i:j]
                y_predict = self.model.forward(x_train_data)
                params, grads = self.model.backward(y_predict, y_train_data)
                self.update(params, grads)
            if self.switch_print and epoch % self.iter_times_print == 0:
                lost_mean = self.model.evaluate_loss(x_train, y_train)
                print('epoch:%d, loss:%.5f' % (epoch, lost_mean))


_dict = {'bgd': BGD,
         }


def get(name):
    val = _dict.get(name, None)
    if val == None:
        raise Exception(('the calculate unit %s is not existed.' % name))
    return val
