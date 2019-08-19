#1. 데이터
import numpy as np #from numpy import array 사용 시, numpy(np) 안써도 가능

# x_train = np.arrange(1, 101, 1)
# x_train = np.array(range(1, 101))
x = np.array([range(100), range(311, 411), range(100)])
y = np.array([range(501, 601), range(711, 811), range(100)])
# x_train = np.array([1,2,3,4,5,6])
# y_train = np.array([1,2,3,4,5,6])
# x_test = np.array([4,5,6])
# y_test = np.array([4,5,6])
# x_val = np.array([1,2,3])
# y_val = np.array([1,2,3])
# print(x_train)

#전치행렬(행과 열을 뒤바꾸어 열값을 활용)
x = np.transpose(x)
y = np.transpose(y)
# x = x.reshape((x.shape[0], x.shape[1], 1))
# y = y.reshape((y.shape[0], y.shape[1], 1))


#Scaler
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# scaler = StandardScaler() #또는 MinMaxScaler
# scaler.fit(x)
# x_scaled = scaler.transform(x)
# print(x_scaled)

#데이터 나누기
#(train 60%, test 40%)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.4 #전체 데이터(x, y)일 경우
)

#test값을 반 쪼개서 val에 제공함으로서 train 60%, test 20%, val 20%
x_test, x_val, y_test, y_val = train_test_split(
    x_test, y_test, random_state=66, test_size=0.5
)


#2. 모델 구성
from keras.callbacks import TensorBoard
td_hist = TensorBoard(
            log_dir='./graph', histogram_freq=0, #import 해준 것임 고로
            write_graph=True, write_images=True)
# keras.callbacks.TensorBoard와 TensorBoard는 같은 말

from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Dropout
from keras import regularizers

model = Sequential()

# model.add(LSTM(100, activation='relu', input_shape=(3,1)))
model.add(Dense(10, input_shape=(3, ), activation='relu')) #input_dim도 가능하다
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
# model.add(Dense(10, kernel_regularizer=regularizers.l1(0.01))) #regularizer
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
model.add(Dense(3)) #마지막은 출력부

#함수식 출력
#==============================
# input1 = Input(shape=(3,))
# dense1 = Dense(10, activation='relu')(input1)
# dense2 = Dense(10)(dense1)
# dense3 = Dense(10)(dense2)
# output1 = Dense(10)(dense3)
#==============================

#ensemble은 따로 볼 것

# model.summary() #모델 개요 확인


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')

hist = model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1,
            validation_data=(x_val, y_val), callbacks=[early_stopping, td_hist])


#4. 평가 및 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)

print("acc : ", acc)
print("loss : ", loss)

y_predict = model.predict(x_test)
print(y_predict)

# x_input = array([11,12,13])
# x_input = x_input.reshape((1,3,1))
# yhat = model.predict(x_input)
# print(yhat)


#번외 측정 정확도
#RMSE formula
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# #RMAE formula
# from sklearn.metrics import mean_absolute_error
# def RMAE(y_test, y_predict):
#     return np.sqrt(mean_absolute_error(y_test, y_predict))
# print("RMAE : ", RMAE(y_test, y_predict))

#R2 formula
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

