#1. 데이터
import numpy as np

# x_train = np.arange(1, 101, 1)
# y_train = np.arange(501, 601, 1)
# x_test = np.arange(1001, 1101, 1)
# y_test = np.arange(1101, 1201, 1)
# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)

#행 무시 열 우선
x_train = np.array([1,2,3,4,5,6,7,8,9,10]) #10행 1열
y_train = np.array([1,2,3,4,5,6,7,8,9,10]) #10행 1열
x_test = np.array([11,12,13,14,15,16,17,18,19,20]) #10행 1열
y_test = np.array([11,12,13,14,15,16,17,18,19,20]) #10행 1열
x_val = np.arange(101, 106, 1)
y_val = np.array(range(101, 106))


#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential() #순차적으로, 모델링

#model.add(Dense(5, input_dim = 1, activation = 'relu')) #input_dim: 1개 밀어넣음
model.add(Dense(5, input_shape = (1, ), activation = 'relu')) #input_dim: 1개 밀어넣음
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))
#레이어 5개 이상(입출포함), 노드는 5개이상
# model.summary()


#3. 훈련
# model.compile(loss = 'mse', optimizer = 'adam', metrics=['accuracy'] )
model.compile(loss = 'mse', optimizer = 'adam', metrics=['mse'] )

# model.fit(x, y, epochs=610, batch_size=3)
model.fit(x_train, y_train, epochs=300, batch_size=1,
          validation_data=(x_val, y_val))


#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("acc : ", acc)

y_predict = model.predict(x_test) #x는 훈련시킨 값
print(y_predict)

#RMSE 구하기
from sklearn.metrics import mean_squared_error #레거시한 머신 러닝 중 하나
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

#R2를 음수가 아닌 0.5이하로 만들어보기
#레이어는 인풋과 아웃풋을 포함 5개 이상, 노드는 레이어당 각각 5개 이상
#batch_size = 1, epochs = 100 이상