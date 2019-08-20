#오늘 종가 구하기
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# days  start_price  high_price  low_price  final_price   volume  exchange_rate

# ■■■■■■■■■■■■■■■■■■■■ data load
kospi = pd.read_csv("./data/kospi.csv", encoding="UTF-8")

# print(kospi)
# print(kospi.shape)    # (599, 7)
kospi = np.array(kospi)

x = kospi[1:5,]
# ■■■■■■■■■■■■■■■■■■■■ data cut
# size = 5
# def split_5(seq, size):
#     aaa = []
#     for i in range(len(seq) - size + 1): # i= 0~5 6줗
#         subset = seq[i:(i+size)] # 0~5 : 4~9 1줄씩
#         aaa.append([item for item in subset])
#     # print(type(aaa))
#     return np.array(aaa)

# dataset = split_5(kospi, size)
# print(dataset)
# x = dataset[1:5, 0:4] #(6,4)
# y = dataset[:, 4, ] #(6, )
# x_train = np.reshape(x_train, (len(kospi) - size + 1, 4, 1))

# print(x_train.shape)

# x_test = np.array([[[11],[12],[13],[14]], [[12],[13],[14],[15]],
#                    [[13],[14],[15],[16]], [[14],[15],[16],[17]]])
# y_test = np.array([15,16,17,18])
# print(x_test.shape)
# print(y_test.shape)


x = kospi[1:, 1:5]
y = kospi[1:, 4:5]
# print("x shape: ", x.shape)   # (598, 4)
# print("y shape: ", y.shape)   # (598, 1)

# ■■■■■■■■■■■■■■■■■■■■ train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state = 0)
# print(x_train.shape, x_test.shape)  # (478, 4) (120, 4)
# print(y_train.shape, y_test.shape)  # (478, 1) (120, 1)

# ■■■■■■■■■■■■■■■■■■■■ scale
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print("x_train.shape: ", x_train.shape)
# print("x_test.shape: ", x_test.shape)

x_train = np.dstack([x_train]*3)
x_test = np.dstack([x_test]*3)
#2. 모델 구성
from keras.callbacks import TensorBoard
td_hist = TensorBoard(
            log_dir='./graph', histogram_freq=0, #import 해준 것임 고로
            write_graph=True, write_images=True)
# keras.callbacks.TensorBoard와 TensorBoard는 같은 말

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, Flatten, BatchNormalization
from keras import regularizers

model = Sequential()

model.add(Conv1D(8, kernel_size=8, strides=1, input_shape=(4, 3), padding='valid', activation='relu'))
# model.add(Conv1D(2, kernel_size=3, padding='same', activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))

model.add(Dense(10))

# # model.add(Dense(10, kernel_regularizer=regularizers.l1(0.01))) # regularizer
# # model.add(BatchNormalization())
# # model.add(Dropout(0.2))

# model.summary()

#3. 훈련
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=1000, mode='auto')

hist = model.fit(x_train, y_train, epochs=100, batch_size=8, verbose=1,
                    callbacks=[early_stopping, td_hist])


#4. evaluate & predict
loss, acc = model.evaluate(x_test, y_test, batch_size=8)

print("acc : ", acc)
print("loss : ", loss)

y_predict = model.predict(x_test)
print(y_predict)

# 5. evaluation index
#RMSE formula
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

#R2 formula
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)
