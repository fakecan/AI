import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

a = np.array(range(1, 101))
batch_size = 8
size = 5
def split_5(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1): # i= 0~5 6줗
        subset = seq[i:(i+size)] # 0~5 : 4~9 1줄씩
        aaa.append([item for item in subset])
    # print(type(aaa))
    return np.array(aaa)

dataset = split_5(a, size)
print("====================")
# print(dataset)
print(dataset.shape)

x_train = dataset[:, 0:4]
y_train = dataset[:, 4:5] # = y_train = dataset[:, 4]

print(x_train.shape)
# print(len(x_train))
x_train = np.reshape(x_train, (len(x_train), size-1, 1))

# x_test = x_train * 2
# y_test = y_train * 2
x_test = x_train + 100
y_test = y_train + 100

# print(x_train)
# print(y_train)

print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

# print(x_test[0])
# print(x_test)


#2 모델 구성
import keras
td_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, #import 해준 것임 고로 
                                      write_graph=True, write_images=True)

model = Sequential()
#1배치 작업을 앞에 4는 열, 1은 짜를
model.add(LSTM(256, batch_input_shape=(batch_size,4,1), stateful=True)) #상태 유지 LSTM, 일반 LSTM보다 잘 맞는 편이다
# model.add(LSTM(128, stateful=True))
# model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))                    

# model.add(Dense(128))
# model.add(Dense(256))
# model.add(Dense(32))
# model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))

# model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))

# model.add(Dense(16))
model.add(Dense(16))

# model.add(Dense(128))
# model.add(Dense(64))
# model.add(Dense(128))
# model.add(Dense(32))
# model.add(Dense(32))


# model.add(BatchNormalization())
# model.add(Dense(32))
# model.add(Dropout(0.2))

model.add(Dense(1))
#stateful: 현상 유지, defalut는 False

# model.summary()


model.compile(loss='mse', optimizer='adam', metrics=['mse'])

num_epochs = 20

early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
#shuffle: 그 전에 했던 훈련 상태를 그대로 쓰겠다, fit했을 시 초기화하지 않겠다
for epoch_idx in range(num_epochs):
    print('epochs : ' + str(epoch_idx))
    
    history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size,
                verbose=2, shuffle=False, validation_data=(x_test, y_test),
                    callbacks=[early_stopping])
    model.reset_states() #현재 상태를 한번 리셋해준다. 이것 때문에 초기화되진 않는다
                         #shuffle의 default는 True
mse, _ = model.evaluate(x_train, y_train, batch_size=batch_size)
print("mse : ", mse)

model.reset_states()

y_predict = model.predict(x_test, batch_size=batch_size)

print(y_predict[0:5])


#RMSE 구하기
from sklearn.metrics import mean_squared_error #레거시한 머신 러닝 중 하나
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))


# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

#105~109

# print(history.history.keys())

'''
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('model')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()
'''

#1. predict 
#x_train    y_train
#1,2,3,4    5
#2,3,4,5    6
#...
#96,97,98,99 100
#
#1. mse값을 1 이하로 만들 것 -> Hidden Layer 3개 이상, dropout 또는 BatchNormalization
#2. RMSE 함수 적용 O
#3. R2 함수 적용 O
#4. EarlyStopping 적용 O
#5. tensorboard 적용 
#6. matplotlib 이미지 적용 mse/epochs
