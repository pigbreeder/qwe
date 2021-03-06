from src.layers import *
import src.optimizer
import src.objective
import numpy as np
import src.util
from config.basic import *
import time
class Container(object):
    # print('To check the detail parameter of layer,you can use describe func.')
    pass

class Sequential(Container):
    def __init__(self, layers=None):
        self.layers = []
        if not layers:
            return
        for layer in layers:
            self.add(layer)

    def add(self, layer):
        if not self.layers:
            if layer.input_size == 0:
                raise Exception('must define input data size')
        self.layers.append(layer)

    def compile(self, objective, optimizer='bgd', init_param_method='glorot_normal'):
        self.objective = src.objective.get(objective)
        self.optimizer = src.optimizer.get(optimizer)(self)
        self.init_param_method = init_param_method
        input_size = self.layers[0].input_size
        # 重新构建网络层，得到真实shape
        for idx, layer in enumerate(self.layers):
            layer.build(idx, input_size, init_param_method)
            input_size = layer.output_size

    def evaluate_loss(self, x_train, y_label):
        return self.objective.loss(self.predict(x_train), y_label)
    # 训练
    def fit(self, x_train, y_train, epochs=DEFAULT_EPOCH, learning_rate=DEFAULT_LEARNING_RATE, batch_size=DEFAULT_BATCH_SIZE, reg_lambda=DEFAULE_REG_LAMBDA):
        print("start fit the trains>>>>>>>>>>>>>>>>>>>>>>>>>>")
        self.optimizer.set_param(n_epoch=epochs, learning_rate=learning_rate, batch_size=batch_size, reg_lambda=reg_lambda)
        self.optimizer.iterate(x_train,y_train)
        print("finish fit the trains>>>>>>>>>>>>>>>>>>>>>>>>>>")

    def predict(self, x_test):
        return self.forward(x_test)

    def forward(self, input_data):
        self.input_data = [input_data]
        st = time.time()
        for layer in self.layers:
            # st = time.time()
            input_data = layer.forward(input_data)
            # print(layer.__class__.__name__,'cost time', (time.time()-st))
            self.input_data.append(input_data)
        # print('forward finish>>>>>>>>>>>>>>>>>>>',(time.time()-st))
        return input_data

    def backward(self, y_predict, y_train):
        # 计算损失函数导数
        dA = self.objective.diff(y_predict, y_train)
        layer_len = len(self.layers)
        grads = []
        params = []
        for idx in range(layer_len - 1, -1, -1):
            # to avoid gradient explode and eliminate, we add grad_clicp
            # dA = src.util.grad_clicp(dA)
            cur_layer = self.layers[idx]
            dA, grad_W, grad_b = cur_layer.backward(self.input_data[idx], dA)
            # 将导数放入待优化队列
            if grad_W is not None:
                grads.append((grad_W, grad_b))
                params.append((cur_layer.W, cur_layer.b))
        return params, grads
    def accuracy(self, x_train,y_label):
        y_predict = self.predict(x_train)
        y_predict = np.argmax(y_predict,axis=1)
        m = y_label.shape[0]
        predit_true = y_label[y_label == y_predict].size
        print('predict true accuracy:%f' %(predit_true*1./m))
    def describe(self):
        for layer in self.layers:
            layer.describe()
    def intro(self):
        print("==============================Sequential==============================")
        for layer in self.layers:
            layer.intro()
        print("==============================Sequential==============================")

if __name__ == '__main__':
    pass
