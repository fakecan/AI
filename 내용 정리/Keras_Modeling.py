from keras.models import Sequential
from keras.layers import Dropout, Flatten, BatchNormalization, Dense, LSTM, Conv2D 
import numpy as np
import tensorflow as tf
from keras import regularizers

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Dataset Loading ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
x_data = []
y_data = []

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Data Splitting ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, random_state=0, train_size=0.8, test_size=0.2
)

size = ?
def split_n(seq, size):
    newList = []
    for i in range(len(seq) - size + 1): # i= 0~5 6줗
        subset = seq[i:(i+size)] # 0~5 : 4~9 1줄씩
        newList.append([item for item in subset])
    # print(type(newList))
    return np.array(newList)

dataset = split_n(arraySet, size)

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Data Preprocessing ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler() #또는 MinMaxScaler
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

def data_scaler(train, test):
    for scaler in [MinMaxScaler()]:
        scaler.fit(train)
        train = scaler.transform(train)
        test = scaler.transform(test)
    return train, test
x_train, x_test = data_scaler(x_train, x_test)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Model Design ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
model = Sequential()
model.add(( , input_shape=(), activation=''))
model.add(Dropout())
model.add(Flatten())
model.add(BatchNormalization())
model.add(())

# model.summary()
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Model Compile ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
model.compile(optimizer='', loss='', metrics=[''])

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Model Fit ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=128, mode='auto')

model.fit(x_train, y_train, epochs=128, batch_size=128,
          callbacks=[early_stopping], validation_data=(x_val, y_val))

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Model Evaluate&Predict ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
loss, acc = model.evaluate(x_test, y_test, batch_size=128)
print("Accuracy: ", acc)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE formula
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 formula
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

# #RMAE formula
# from sklearn.metrics import mean_absolute_error
# def RMAE(y_test, y_predict):
#     return np.sqrt(mean_absolute_error(y_test, y_predict))
# print("RMAE : ", RMAE(y_test, y_predict))
