import numpy as np
class Optimizer(object):
    pass

    @staticmethod
    def SGD(x_train, y_train, NN, epoch_num=100, learning_rate=0.01, batch_size=32, eps=1e-8):
        pass
        theta = 0
        sample_size = x_train.shape[0]
        for epoch in range(epoch_num):
            for i in range(0, sample_size, batch_size):
                j = i + batch_size
                if j > sample_size:
                    j = sample_size
                batch_data = x_train[i:j]
                NN.forward(batch_data)
                lost = NN.calc_lost(y_train)
                if lost < eps:
                    return
                NN.backward(lost, learning_rate)
