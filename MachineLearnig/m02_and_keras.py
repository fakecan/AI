import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation

#1. 데이터
x_train = np.array([[0,0], [1,0], [0,1], [1,1]]) # and 모델
y_train = np.array([0,0,0,1])
print(x_train.shape)
print(y_train.shape)

#2. 모델
model = Sequential()

model.add(Dense(8, input_shape=(2, ), activation='relu'))
model.add(Dense(8))
model.add(Dense(1))
# model.add(Activation('sigmoid'))
model.add(Activation('softmax'))

#3. 실행
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=200, batch_size=2)

#4. 평가 및 예측
x_test = np.array([[0,0], [1,0], [0,1], [1,1]])
y_test = np.array([0,0,0,1])
# import math

y_predict = np.round(model.predict(x_test)) # round: 반올림
# y_predict = list(y_predict)
# print(y_predict[1])

loss, acc = model.evaluate(x_test, y_test)
print("예측결과 :\n", y_predict)
print("acc = ", acc)



# x_train = np.reshape(x_train, (x_train.shape[0], 2, 1))
# x_test = np.array([[0,0], [1,0], [0,1], [1,1]])
# y_test = np.array([0,0,0,1])
# x_test = np.reshape(x_test, (x_test.shape[0], 2, 1))

# x_train = np.reshape(x_train, (x_train.shape[0], 2, 1))
# #2. 모델
# model = Sequential()

# model.add(LSTM(8, input_shape=(2, 1), activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1))
# model.add(Activation('softmax'))

# #3. 실행
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=100, batch_size=2)

# #4. 평가 및 예측
# x_test = np.array([[0,0], [1,0], [0,1], [1,1]])
# y_test = np.array([0,0,0,1])
# x_test = np.reshape(x_test, (x_test.shape[0], 2, 1))

# y_predict = model.predict(x_test)

# loss, acc = model.evaluate(x_test, y_test, batch_size=2)
# print("예측결과 :\n", y_predict)
# print("acc = ", acc)
# # accuracy_score(y_data, y_predict)

