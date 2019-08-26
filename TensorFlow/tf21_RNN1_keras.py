from keras.models import Sequential
from keras.layers import Dropout, Flatten, BatchNormalization, Dense, LSTM
import numpy as np
import tensorflow as tf

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Dataset Loading ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
idx2char = ['e', 'h', 'i', 'l', 'o']    # Alphabet Sequences

_data = np.array([['h', 'i', 'h', 'e', 'l', 'l', 'o']], dtype=np.str).reshape(-1, 1)
# print(_data.shape)  # (7, 1)
# print(_data)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray().astype('float32')    # float64에서 float32로 변환
# print(_data.shape)  # (7, 5)
# print(_data)

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Data Splitting ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
x_data = _data[:6, ]    # (6, 5)
y_data = _data[1:, ]    # (6, 5)
# y_data = np.argmax(y_data, axis=1)
# print(x_data.shape, y_data.shape)   # (6, 5) (6,)
# print(x_data, y_data)

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Data Reshape ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
x_data = x_data.reshape(1, 6, 5)    # (1, 6, 5)
y_data = y_data.reshape(1, 6*5)    # (1, 6, 5)
# print(x_data.shape, y_data.shape)   # (6, 5) (6,)
# print(x_data, y_data)

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(
#     x_data, y_data, random_state=0, train_size=0.8, test_size=0.2
# )
# print(x_train.shape, x_test.shape)  # (4, 5) (2, 5)
# print(y_train.shape, y_test.shape)  # (4,) (2,)

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Model Design ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
model = Sequential()
model.add(LSTM(32, input_shape=(6, 5), return_sequences=True))
# model.add(LSTM(64, input_shape=(6, 5), return_sequences=True))
model.add(LSTM(16))
# model.add(Dense(32, activation='relu'))
model.add(Dense(6*5, activation='softmax'))

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Model Compile ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Model Fit ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
model.fit(x_data, y_data, epochs=100, batch_size=1)

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Model Evaluate&Predict ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# loss, acc = model.evaluate(x_data, y_data, batch_size=1)
# print("Accuracy: ", acc)

y_predict = model.predict(x_data)
y_predict = y_predict.reshape(6, 5)
print(y_predict)

y_predict = np.argmax(y_predict, axis=1)
result_str = [idx2char[c] for c in np.squeeze(y_predict)]
print(result_str)
