import numpy as np
class LostFunction(object):
    @staticmethod
    def linear(AL, Y):
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(np.square(np.subtract(AL,Y)))
        return cost
    @staticmethod
    def linear_derivative(AL, Y):
        """

        :param y_predit:
        :param y_label:
        :return:
        """
        pass
        return AL - Y

    @staticmethod
    def cross_entropy(AL,Y):
        m = Y.shape[0]
        cost_mean = (1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
        cost = np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL))
        return cost, cost_mean
    @staticmethod
    def cross_entropy_derivative(AL, Y):
        """
        C(x) = y * log(h(x)) + (1 - y) * log(1 - h(x))
        dC/dh = y-h
        :param y_predit:
        :param y_label:
        :return:
        """
        # return Y * 1./AL + (Y - 1) * (1 - AL)
        return Y * np.divide(1.,AL) + (Y - 1) * (1 - AL)

if __name__ == '__main__':
    np.random.seed(10)
    AL = np.random.rand(2,4)
    Y = np.random.rand(2,4)
    print(LostFunction.cross_entropy(AL,Y))
    print(LostFunction.cross_entropy_derivative(AL,Y))
