import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Flatten
from keras.callbacks import EarlyStopping
import datetime

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Dataset Loading ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
df = pd.read_csv('./data/test0822.csv', encoding='UTF-8', sep=',')
# print(df.shape)   # (5479, 9)
# print(df)   # date  kp_0h  kp_3h  kp_6h  kp_9h  kp_12h  kp_15h  kp_18h  kp_21h
# print(df.head(), "\n\n", df.tail())
df = df.dropna(how='any')   # empty low delete
# print(df.shape)     # (5474, 9)      
dataset = np.array(df)

# ■■■■■■■■■■ 날짜를 0~364로 변환 ■■■■■■■■■■
def day_change(seq):
    arr = []
    for i in seq[:,0]:
        d = datetime.datetime.strptime(i, '%Y-%m-%d')
        y = str(d.year)+'-1-1'
        y = datetime.datetime.strptime(y, '%Y-%m-%d')
        sub = d - y
        sub = sub.days
        arr.append(sub)
    seq[:,0] = arr
    return seq

dataset = day_change(dataset)
# print(dataset)

# ■■■■■■■■■■ Data Split ■■■■■■■■■■
size = 10
def split_n(seq, size):
    newList = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        newList.append([item for item in subset])
    # print(type(newList))
    return np.array(newList)

dataset = split_n(dataset, size)
# print(dataset.shape)    # (5465, 10, 9)
# print(dataset)
# np.save('./WeekTest/tf_test0822.txt', dataset)

x_data = dataset[:, :5, :]
y_data = dataset[:, 5:, 1:]
# print(x_data)
# print(y_data)
# print(x_data.shape, y_data.shape)   # (5465, 5, 9) (5465, 5, 8)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size = 0.2, random_state = 66, shuffle=False
)
print(x_train.shape, y_train.shape) # (4372, 5, 9) (4372, 5, 8)
print(x_test.shape, y_test.shape)   # (1093, 5, 9) (1093, 5, 8)

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

y_train = y_train.reshape(-1, 5*8)
y_test = y_test.reshape(-1, 5*8)

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Model Design ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
model = Sequential()

model.add(LSTM(256, input_shape=(5, 9), return_sequences=True))
model.add(LSTM(64))
model.add(Dense(96, activation='relu'))
model.add(Dense(5*8))

# model.summary()

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Model Compile ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Model Fit ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=4, verbose=1,
          callbacks=[early_stopping])

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Model Evaluate & Predict ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
question = np.array([['2007-07-06',1,1,1,1,2,2,1,2], ['2007-07-07',2,1,1,1,1,2,2,2],
                     ['2007-07-08',1,2,1,1,1,1,1,1], ['2007-07-09',1,1,0,0,1,1,1,1],
                     ['2007-07-10',1,0,0,1,1,2,1,3]])
question = day_change(question)
question = question.reshape(1, 5, 9)
# question = question.astype(np.float64)

y_predict = model.predict(question)
print('y_predict: \n', y_predict)

y_predict = np.round(y_predict)
print('------------------------------')
print('y_predict(round):', y_predict)

y_predict = model.predict(x_test)
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
