import numpy as np
class Optimizer(object):
    pass

    @staticmethod
    def BGD(x_train, y_train, NN, epoch_num=100, learning_rate=0.001, batch_size=32, eps=1e-8):
        pass
        theta = 0
        sample_size = x_train.shape[0]
        for epoch in range(epoch_num):
            # print('epoch:',epoch)
            for i in range(0, sample_size, batch_size):
                j = i + batch_size
                if j > sample_size:
                    j = sample_size
                x_train_data = x_train[i:j]
                y_train_data = y_train[i:j]
                NN.forward(x_train_data)
                lost, lost_mean = NN.calc_lost(y_train_data)
                print(lost_mean)
                if np.abs(lost_mean) < eps or lost_mean is np.nan:
                    return
                NN.backward(y_train_data, learning_rate) # careful
