import numpy as np
class Optimizer(object):
    pass

    @staticmethod
    def BGD(x_train, y_train, NN, epoch_num=100, learning_rate=0.001, batch_size=32, eps=1e-4):
        pass
        theta = 0
        sample_size = x_train.shape[0]
        for epoch in range(epoch_num + 1):
            # print('epoch:',epoch)
            for i in range(0, sample_size, batch_size):
                j = i + batch_size
                if j > sample_size:
                    j = sample_size
                x_train_data = x_train[i:j]
                y_train_data = y_train[i:j]
                NN.forward(x_train_data)
                NN.backward(y_train_data, learning_rate) # careful
            if epoch % 100 == 0:
                lost, lost_mean = NN.calc_lost(x_train, y_train)
                print('epoch:%d, loss:%.5f' % (epoch, lost_mean))