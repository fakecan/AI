#오늘 종가 구하기
import csv
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt

f = open('./data/kospi.csv', 'r')
dataset = list(csv.reader(f))
# print(dataset)
# for line in dataset:
#     print(line)
# f.close()

ds = np.array(dataset)

print(ds)
print("ds.shape: ", ds.shape)

'''
x_train = ds[1:, 1:4]
y_train = ds[1:, 4:5]
# print("x: \n", x)
# print("y: \n", y)
# print("x.shape: ", x.shape)
# print("y.shape: ", y.shape)

#train 60프로 test 40프로
# from sklearn.model_selection import train_test_split #비율 맞춰 잘라주는 함수 #열 분할
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, random_state = 66, test_size = 0.4
# )

# #test와 val이 절반씩 나눠가짐
# x_test, x_val, y_test, y_val = train_test_split(
#     x_test, y_test, random_state = 66, test_size = 0.5
# )
x_test = np.array([[2036.46, 2035.32, 2059.13, 2063.35, 2085.67],
                   [2041.16, 2044.59, 2063.13, 2068.16, 2088.81],
                   [2010.95, 2032.61, 2025.01, 2054.64, 2061.08]])
y_test = np.array([2024.55, 2038.68, 2029.48, 2066.26, 2074.48])
# x_test = np.array([2036.46, 2041.16, 2010.95])
# y_test = np.array([2024.55])

x_test = np.transpose(x_test)

scaler = StandardScaler() #또는 MinMaxScaler
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
# x_test_scaled = scaler.transform(x_test)
# print(x_scaled)

# print("x_train.shape: ", x_train.shape)
# print("x_test.shape: ", x_test.shape)
# x_train = x_train.reshape(x_train, (len(x_train), 3, 1))
# x_test = x_test.reshape(x_train, (len(x_test), 3, 1))


#2. 모델 구성
from keras.callbacks import TensorBoard
td_hist = TensorBoard(
            log_dir='./graph', histogram_freq=0, #import 해준 것임 고로
            write_graph=True, write_images=True)
# keras.callbacks.TensorBoard와 TensorBoard는 같은 말

from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import regularizers

model = Sequential()

# model.add(Conv2D(32, kernel_size=(3,3), input_shape=(3,1,1), activation='relu'))
# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.25))
# model.add(Flatten()) #데이터를 평평하게 편다.
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))


model.add(Dense(128, activation='relu', input_shape=(3, )))
model.add(Dense(128, activation='relu')) #input_dim도 가능하다
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))


model.add(Dense(1)) #마지막은 출력부

# # model.add(Dense(10, kernel_regularizer=regularizers.l1(0.01))) #regularizer
# # model.add(BatchNormalization())
# # model.add(Dropout(0.2))

# model.summary()



#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=1000, mode='auto')

hist = model.fit(x_train, y_train, epochs=100, batch_size=8, verbose=1,
                    callbacks=[early_stopping, td_hist])


#4. 평가 및 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=8)

print("acc : ", acc)
print("loss : ", loss)

y_predict = model.predict(x_test)
print(y_predict)


#번외 측정 정확도
#RMSE formula
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

#R2 formula
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

'''