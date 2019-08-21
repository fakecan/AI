from keras.datasets import mnist
from keras.applications import ResNet50
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, array_to_img
from keras.utils import to_categorical
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape)    # (60000, 28, 28)
# print(x_test.shape)     # (10000, 28, 28)
# print(y_train.shape)    # (60000,)
# print(y_test.shape)     # (10000,)

x_train = np.array(x_train).reshape((-1,np.prod(x_train.shape[1:])))
x_test = np.array(x_test).reshape((-1,np.prod(x_train.shape[1:])))
# x_train = x_train.reshape(60000, 28*28)     # (60000, 784)
# x_test = x_test.reshape(10000, 28*28)       # (10000, 784)
# print(x_train.shape)
# print(x_test.shape)

# ■■■■■■■■■■ Convert add 3 channel 
x_train = np.dstack([x_train]*3)
x_test = np.dstack([x_test]*3)
# print(x_train.shape)    # (60000, 784, 3)
# print(x_test.shape)     # (10000, 784, 3)

# ■■■■■■■■■■ Reshape
x_train = x_train.reshape(-1, 28, 28, 3)
x_test = x_test.reshape(-1, 28, 28, 3)
# print(x_train.shape)    # (60000, 28, 28, 3)
# print(x_test.shape)     # (10000, 28, 28, 3)

# ■■■■■■■■■■ Resize
x_train = np.asarray([img_to_array(array_to_img(
                      im, scale=False).resize((64, 64))) for im in x_train])
x_test = np.asarray([img_to_array(array_to_img(
                     im, scale=False).resize((64, 64))) for im in x_test])

# ■■■■■■■■■■ Scale                     
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# ■■■■■■■■■■ Categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

conv_base = ResNet50(weights='imagenet', include_top=False,
                  input_shape=(64, 64, 3))

model = Sequential()
model.add(conv_base)

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(10, activation='softmax'))
# model.summary()

model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=10, batch_size=64)

print("acc: ",model.evaluate(x_test,y_test)[1])