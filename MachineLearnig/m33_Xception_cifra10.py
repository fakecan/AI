from keras.datasets import cifar10
from keras.applications import Xception
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, array_to_img
from keras.utils import to_categorical
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape)    # (50000, 32, 32, 3)
# print(x_test.shape)     # (10000, 32, 32, 3)

# ■■■■■■■■■■ Scale                     
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# ■■■■■■■■■■ Resize
x_train = np.asarray([img_to_array(array_to_img(
                      im, scale=False).resize((80, 80))) for im in x_train])
x_test = np.asarray([img_to_array(array_to_img(
                     im, scale=False).resize((80, 80))) for im in x_test])

# ■■■■■■■■■■ Categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

conv_base = Xception(weights='imagenet', include_top=False,
                  input_shape=(80, 80, 3))

model = Sequential()
model.add(conv_base)

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))

model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))

model.add(Dense(10, activation='softmax'))
# model.summary()

model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=10, batch_size=64)

print("Accuracy : ",model.evaluate(x_test,y_test)[1])