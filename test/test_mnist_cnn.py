from test.parse_mnist import *
from src.container import *
from src.layers import *
train_images = load_train_images(num_data=1000) / 255.
train_images = train_images.reshape(*(train_images.shape),1)
train_labels = load_train_labels(num_data=1000)

test_images = load_test_images(num_data=10) / 255
test_images = test_images.reshape(*(test_images.shape),1)
test_labels = load_test_labels(num_data=10)
model = Sequential()
model.add(Convolution2D(5,(5,5),input_size=(28,28,1)))
model.add(Activation('relu'))
model.add(Pool((2,2)))
model.add(Convolution2D(5, (5,5)))
model.add(Activation('relu'))
model.add(Pool((2,2)))
model.add(Flatten())
model.add(Dense(200))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(objective='CategoricalCrossEntropy',init_param_method='glorot_normal')
model.intro()
print('start fit')
model.fit(train_images,train_labels,epochs=20,learning_rate=0.02)

y_test = model.predict(test_images)
model.accuracy(test_images,test_labels)
# print(y_test)
# print(test_labels)

