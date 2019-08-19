#1. 데이터
import numpy as np

x = np.array(range(1, 101))
y = np.array(range(1, 101))
#print(x)
#print(y)
#1~60까지 train 61~80 validation 81~100 test
# x_train = x[:60]
# y_train = y[:60]
# x_val = x[60:80]
# y_val = y[60:80]
# x_test = x[80:]
# y_test = y[80:]

# print(x_train)
# print(y_train)
# print(x_validation)
# print(y_validation)
# print(x_test)
# print(y_test)

#train 60프로 test 40프로
from sklearn.model_selection import train_test_split #비율 맞춰 잘라주는 함수
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state = 66, test_size = 0.4
)
# print(x_test)
# print(y_test)

#test와 val이 절반씩 나눠가짐
x_test, x_val, y_test, y_val = train_test_split(
    x_test, y_test, random_state = 66, test_size = 0.5
)

# print(x_test)
# print(y_test)


# #2. 모델 구성
# from keras.models import Sequential
# from keras.layers import Dense
# model = Sequential() #순차적으로, 모델링

# model.add(Dense(5, input_dim = 1, activation = 'relu')) #input_dim: 1개 밀어넣음
# # model.add(Dense(5, input_shape = (1, ), activation = 'relu')) #input_dim: 1개 밀어넣음
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(1))
# #레이어 5개 이상(입출포함), 노드는 5개이상
# # model.summary()


# #3. 훈련
# # model.compile(loss = 'mse', optimizer = 'adam', metrics=['accuracy'] )
# model.compile(loss = 'mse', optimizer = 'adam', metrics=['mse'] )

# # model.fit(x, y, epochs=610, batch_size=3)
# model.fit(x_train, y_train, epochs=100, batch_size=1,
#           validation_data=(x_val, y_val))


# #4. 평가 예측
# loss, acc = model.evaluate(x_test, y_test, batch_size=1)
# print("acc : ", acc)

# y_predict = model.predict(x_test) #x는 훈련시킨 값
# print(y_predict)

# #RMSE 구하기
# from sklearn.metrics import mean_squared_error #레거시한 머신 러닝 중 하나
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# print("RMSE : ", RMSE(y_test, y_predict))

# # R2 구하기
# from sklearn.metrics import r2_score
# r2_y_predict = r2_score(y_test, y_predict)
# print("R2 : ", r2_y_predict)