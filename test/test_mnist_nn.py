from test.parse_mnist import *
from src.container import *
from src.layers import *
train_images = load_train_images(num_data=6000) / 255.
train_images = train_images.reshape(6000,-1)
train_labels = load_train_labels(num_data=6000)

test_images = load_test_images(num_data=100) / 255.
test_images = test_images.reshape(100,-1)
test_labels = load_test_labels(num_data=100)
print(train_images.shape)
model = Sequential()
model.add(Dense(100,input_size=784))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(objective='CategoricalCrossEntropy',init_param_method='glorot_normal')
print('start fit')
model.fit(train_images,train_labels,epochs=200,learning_rate=0.01)
# y_test = model.predict(test_images)
# print(y_test)
# print(test_labels)