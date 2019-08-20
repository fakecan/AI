import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

tf.set_random_seed(777)

xy = np.loadtxt('./data/data-04-zoo.csv', delimiter=',', dtype=np.float32)  # numpy: 숫자만 인식
print(xy.shape)   # (101, 17)

x_data = xy[: , 0:-1]
y_data = xy[: , [-1]]
y_data = np_utils.to_categorical(y_data)
print(x_data.shape, y_data.shape)   # (101, 16) (101, 7)

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, random_state = 0, test_size = 0.2
)

print(x_train.shape, x_test.shape)  # (80, 16) (21, 16)
print(y_train.shape, y_test.shape)  # (80, 7) (21, 7)

# model
model = Sequential()

model.add(Dense(7, input_shape=(16, ), activation='softmax'))

# fit
early_stopping = EarlyStopping(monitor='acc', patience=10000, mode='auto') #monitor loss acc
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=1, callbacks=[early_stopping])

# evaluate & predict
_, acc = model.evaluate(x_test, y_test, batch_size=1)
y_predict = model.predict(x_test)

print("Accuracy : ", acc)
# print('Predict : ', y_predict)
print("Predict : ", np.argmax(y_predict, axis=1))