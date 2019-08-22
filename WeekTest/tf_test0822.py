import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Flatten
from keras.callbacks import EarlyStopping
from pandas.tseries.offsets import YearEnd

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Dataset Loading ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
df = pd.read_csv('./data/test0822.csv', encoding='UTF-8', sep=',')
# print(df.shape)    # (5479, 9)
# print(df)

# print(df.head())
# print(df.tail())
# df = df.dropna(how='any')

df['date'] = pd.to_datetime(df['date']) + YearEnd(1)
df = df.set_index('date')
# df.plot()
# plt.show()

split_date = pd.Timestamp('2011-01-01')
# 이전은 train, 이후는 test
train = df.loc[:split_date, ['kp_0h', 'kp_3h', 'kp_6h', 'kp_9h',
                             'kp_12h', 'kp_15h', 'kp_18h', 'kp_21h']]
test = df.loc[split_date:, ['kp_0h', 'kp_3h', 'kp_6h', 'kp_9h',
                            'kp_12h', 'kp_15h', 'kp_18h', 'kp_21h']]
print(train.shape, test.shape)
# print(train)
# print(test)

# ax = train.plot()
# test.plot(ax=ax)
# plt.legend(['train', 'test'])
# plt.show()

# train = np.array(train)
# test = np.array(test)
# print(type(train), type(test))
# print(train.shape, test.shape)  # (4382, 8) (1097, 8)

x_train, y_train, x_test, y_test = train_test_split(
    train, test, test_size = 0.2, random_state = 66)


# x_test, y_test = train_test_split(
#     test, train_size=0.5, shuffle=False, random_state=66)
print(x_train.shape, x_test.shape)  # (877, 8) (220, 8)
print(y_train.shape, y_test.shape)  # (3505, 8, 1) (877, 8, 1)

'''
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 8, 1)
x_test = x_test.reshape(-1, 8, 1)
print(x_train.shape, x_test.shape)


# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Model Design ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
model = Sequential()

model.add(LSTM(256, input_shape=(8, 1), return_sequences=True))
model.add(LSTM(64))

# model.add(LSTM(64, return_sequences=True))
# model.add(LSTM(8))
# model.add(Flatten())
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
model.add(Dense(8))

# model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')

model.fit(x_train, y_train, epochs=10, batch_size=8, verbose=1,
          callbacks=[early_stopping])

y_predict = model.predict(x_test)
print('y_predict(x_test) : \n', y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

'''