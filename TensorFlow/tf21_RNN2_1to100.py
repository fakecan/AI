# 1 ~ 100까지의 숫자를 이용해서 6개씩 잘라서 RNN 구성
# train, test로 분리

# 1, 2, 3, 4, 5, 6 : 7
# 2, 3, 4, 5, 6, 7 : 8
# 3, 4, 5, 6, 7, 8 : 9
# ...
# 94, 95, 96, 97, 98, 99 : 100

# predict 101 ~ 110까지 예측하시오.
# 지표 RMSE

from keras.models import Sequential
from keras.layers import Dropout, Flatten, BatchNormalization, Dense, LSTM
import numpy as np

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Dataset Loading ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
arraySet = np.array(range(1,101))

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Data Splitting ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
size = 7
def split_n(seq, size):
    newList = []
    for i in range(len(seq) - size + 1): # i= 0~5 6줗
        subset = seq[i:(i+size)] # 0~5 : 4~9 1줄씩
        newList.append([item for item in subset])
    # print(type(newList))
    return np.array(newList)

dataset = split_n(arraySet, size)
# print(dataset)
# print(dataset.shape)    # (94, 7)
x_data = dataset[:, 0:6] #(6)
y_data = dataset[:, -1] #(6, )
# print(x_data, "\n", y_data)
# print(x_data.shape, y_data.shape)   # (94, 6) (94,)
y_data = y_data.reshape(-1, 1)
# print(y_data.shape)   # (94, 1)

# x_data = x_data.reshape(1, 94, 6)
# y_data = y_data.reshape(1, 94, 1)
# print(x_data.shape, y_data.shape)   #


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, random_state=66, test_size=0.5, train_size=0.5,
)
# print(x_train.shape, x_test.shape)  # (75, 6) (19, 6)
# print(y_train.shape, y_test.shape)  # (75, 1) (19, 1)

x_train = x_train.reshape(-1, 1, 6)
x_test = x_test.reshape(-1, 1, 6)
# print(x_train.shape, x_test.shape)  # (47, 1, 6) (47, 1, 6)
# print(x_train, "\n", x_test)

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Model Design ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
model = Sequential()

model.add(LSTM(32, input_shape=(1, 6), activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Model Compile ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Model Fit ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
model.fit(x_train, y_train, epochs=100, batch_size=1)

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Model Evaluate&Predict ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("Accuracy: ", acc)

# y_predict = model.predict(x_test)
# print(y_predict)

x_input = np.array(range(95, 111))
x_input = split_n(x_input, 7)
x_input = x_input[:, :6]
x_input = x_input.reshape(-1, 1, 6) #1행 3열. 1,3, ?
y_predict = model.predict(x_input, verbose=1)
print(y_predict)
# print(x_input)
# print(x_input.shape)

aw = np.array(range(101, 111))
aw = aw.reshape(10, 1)

# RMSE solution
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(aw, y_predict))
