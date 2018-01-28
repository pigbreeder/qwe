from abc import ABCMeta, abstractmethod
from config import *
import numpy as np
import time

class Optimizer(object):
    __metaclass__ = ABCMeta

    def __init__(self, model):
        pass
        self.model = model

    def set_param(self, n_epoch, learning_rate, batch_size, reg_lambda, switch_print=SWITCH_PRINT_TRAIN_MSG, iter_times_print=DEFAULT_LOSS_PRINT_LOOP):
        self.learning_rate = learning_rate
        self.iterations = 0
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.reg_lambda = reg_lambda
        self.switch_print = switch_print
        self.iter_times_print = iter_times_print

    @abstractmethod
    def update(self, params, grads):
        pass

    @abstractmethod
    def iterate(self, x_train, y_train):
        pass


class BGD(Optimizer):

    def __init__(self, model, momentum=0.9, decay=1e-6):
        super().__init__(model)
        self.momentum = momentum
        self.decay = decay

        self.pre_velocities = []
        self.now_velocities = []



    def update(self, params, grads):
        super().update(params, grads)


        def debug_print():
            # print('epoch:',self.iterations)
            # print('grads:', grads)
            #
            # for idx, layer in enumerate(self.model.layers):
            #     print('idx:',idx)
            #     print('w:',layer.W)
            #     print('b:',layer.b)
            # print('========================')
            pass
        # for param, grad in zip(params, grads):
        #     pw, pb = param
        #     gw, gb = grad
        #
        #     gw += self.reg_lambda * pw
        #     pw -= self.learning_rate * gw
        #     pb -= self.learning_rate * gb

        if self.iterations == 0:
            self.pre_velocities = [0] * len(params)
        # 将元组类型展开便于计算
        params = [param for layer_param in params for param in layer_param]
        grads = [grad for layer_grad in grads for grad in layer_grad]
        # 梯度下降速度控制
        for param, grad, pre_velocity in zip(params, grads, self.pre_velocities):
            now_velocity = - self.cur_lr * grad + pre_velocity * self.momentum
            param += now_velocity
            self.now_velocities.append(now_velocity)

        debug_print()
        self.preset_iterate()
        return

    def preset_iterate(self):
        self.iterations += 1
        self.cur_lr = self.learning_rate * 1. / (1.0 + self.decay * self.iterations)
        self.pre_velocities = self.now_velocities
        self.now_velocities = []

    def iterate(self, x_train, y_train):
        pass
        self.st_time = time.time()
        self.cur_lr = self.learning_rate
        sample_size = x_train.shape[0]
        for epoch in range(1, self.n_epoch + 1):
            st = time.time()
            for i in range(0, sample_size, self.batch_size):
                j = i + self.batch_size
                if j > sample_size:
                    j = sample_size
                # 获取每次样本
                x_train_data = x_train[i:j]
                y_train_data = y_train[i:j]
                y_predict = self.model.forward(x_train_data)
                params, grads = self.model.backward(y_predict, y_train_data)
                self.update(params, grads)
            if self.switch_print and epoch % self.iter_times_print == 0:
                lost_mean = self.model.evaluate_loss(x_train, y_train)
                print('epoch:%d, loss:%.5f,cost time:%.2f' % (epoch, lost_mean,time.time()-st))
        print('finish iterate,cost %.2f time.' % (time.time()-self.st_time))

class BGD_REG(BGD):
    # 带有正则约束，减少一定过拟合
    def update(self, params, grads):
        for param, grad in zip(params, grads):
            pw, pb = param
            gw, gb = grad

            gw += self.reg_lambda * pw
            pw -= self.learning_rate * gw
            pb -= self.learning_rate * gb


_dict = {'bgd': BGD,
         'bgd_reg': BGD_REG,
         }


def get(name):
    val = _dict.get(name, None)
    if val == None:
        raise Exception(('the calculate unit %s is not existed.' % name))
    return val
