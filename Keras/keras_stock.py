import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
import pandas as pd
import os
import csv
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt


#일자 | 시가 | 고가 | 저가 | 종가 | 거래량 | 환율
#1. 데이터
f = open('D:\\study\\keras\\kospi200test.csv', 'r')
dataset = list(csv.reader(f))
ds = np.array(dataset)
print(ds)
print("ds.shape: ", ds.shape)
# x_train = ds[1:, 1:4]
# y_train = ds[1:, 4:5]
# print(x_train)
# print(y_train)


size = 3
def split_5(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1): # i= 0~5 6줗
        subset = seq[i:(i+size)] # 0~5 : 4~9 1줄씩
        aaa.append([item for item in subset])
    # print(type(aaa))
    return np.array(aaa)

splset = split_5(ds, size)
print(splset.shape)
x_train = splset[1:, 1:4]
# y_train = splset[:, 4:5]
print(x_train)
# print(y_train)
# print(x_train.shape)
# print(y_train.shape)

'''
x_train = np.reshape(x_train, (x_train.shape[0], 2, 2))
print(x_train.shape)

x_test = np.array(range(47,54))
y_test = np.array(range(51,58))
x_test = split_5(x_test, 4)
y_test = split_5(y_test, 4)
print(x_test)
print(y_test)

x_test = np.reshape(x_test, (x_test.shape[0], 2, 2))
print(x_test.shape)


#
#2. 모델 구성
model = Sequential()

model.add(LSTM(32, input_shape=(3,3), activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16))
model.add(Dense(1))

# model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')

model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1,
          callbacks=[early_stopping])

loss, acc = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

print('y_predict(x_test) : \n', y_predict)
print('loss : ', loss)
print('acc : ', acc)
'''