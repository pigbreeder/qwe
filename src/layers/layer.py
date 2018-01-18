from abc import ABCMeta, abstractmethod
import numpy as np
class Layer(object):
    __metaclass__ = ABCMeta

    def __init__(self, output_size, input_size=0, name=''):
        self.input_size = input_size
        self.output_size = output_size
        self.name = name
        self.W = None
        self.b = None
    @abstractmethod
    def build(self, loc_idx, input_size, init_param_method):
        self.loc_idx = loc_idx
        self.input_size = input_size
        pass
    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, X, dA):
        pass

    def set_param(self, W, b):
        self.W = W
        self.b = b

    def describe(self):
        pass
        print('the layer ' + self.name + '\'s W\n\t' + str(self.W))
        print('the layer ' + self.name + '\'s b\n\t' + str(self.b))


    def intro(self):
        print('%s layer\tname:%s\ttype:%s' % (self.loc_idx, self.name, self.__class__.__name__))
        print('input_size:%s\toutput_size:%s' % (self.input_size, self.output_size))


if __name__ == '__main__':
    l = Layer(3,name='test')
    l.intro()
    l.describe()
