import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1,101))

size = 8
def split_5(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1): # i= 0~5 6줗
        subset = seq[i:(i+size)] # 0~5 : 4~9 1줄씩
        aaa.append([item for item in subset])
    # print(type(aaa))
    return np.array(aaa)

dataset = split_5(a, size)
print(dataset.shape)
x_train = dataset[:, 0:4]
y_train = dataset[:, 4:8]
print(x_train.shape)
print(y_train.shape)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

x_test = np.array(range(47,54))
y_test = np.array(range(51,58))
x_test = split_5(x_test, size // 2)
y_test = split_5(y_test, size // 2)
print(x_test)
print(y_test)

x_test = np.reshape(x_test, (x_test.shape[0], 4, 1))
print(x_test.shape)


#'''
#2. 모델 구성
model = Sequential()

model.add(LSTM(32, input_shape=(4,1), activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10))
model.add(Dense(4))

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
# '''