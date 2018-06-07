# qwe
[相关文章解析-intro](http://www.cnblogs.com/pigbreeder/p/8375935.html)

[相关文章解析-CNN实现](http://www.cnblogs.com/pigbreeder/p/8376034.html)

---

## 简介
简单的深度框架，参考Ng的深度学习课程作业，使用了keras的API设计。

方便了解网络具体实现，避免深陷于成熟框架的细节和一些晦涩的优化代码。

网络层实现了Dense, Flatten, Convolution2D, Activation, Dropout等。

优化算法实现了带有动量的梯度下降，同时还有一个带正则的优化算法备选。

权重初始化有多维高斯分布和Xavier initialization两种。

目标函数有MSE，CategoricalCrossEntropy。

在测试中有全连接网络与CNN手写字体识别示例。

## 环境与安装

python 3.x

**依赖**
1. numpy
2. sklearn
2. matplotlib
2. numba

**安装**

**无需安装，使用前请先设置PYTHONPATH路径**

为`qwe`工程根目录

> eg.qwe位于/home/test/qwe, 执行 export PYTHONPATH=$PYTHONPATH:/home/test/qwe

在config/basic.py中通过SWITCH_EXT选择是否开启扩展，默认关闭。
> 若选择开启，在qwe/src/ext目录下执行 python setup.py build_ext -i 即可

**测试**

在test目录下执行 `python testfile.py` 即可

## 目录结构

1. src/
    1. ext/
        1. src/
            1. convUtil.pyx Cython的扩展，加速CNN
    2. layers/
        1.activation.py 激活函数,sigmoid, ReLU, tanh
        1.convolution2D.py 卷积层
        1.simple_convolution2D.py 简单卷积未优化
        1.dense.py   全连接层
        1.dropout.py 丢弃层？
        1.flatten.py 拉伸层
        1.pool.py 池化层,max average
    3. container.py  模型容器，equential
    3. initialization.py 参数初始化方法
    3. objective.py 目标函数 
    3. optimizer.py 优化方法
    3. py_util.py py 实现的方法
    3. unit.py 计算单元
    3. util.py 一些方法
2. test/
    1. parse_mnist.py 解析mnist图片
    1. test_col2img.py 测试
    1. test_mnist_cnn.py 使用CNN训练 mnist
    1. test_mnist_nn.py 使用全连接训练 mnist
    1. test_nn.py 使用全连接训练sklearn一个数据集
    1. test_objective.py 测试
    1. test_unit.py 测试


## 待续
