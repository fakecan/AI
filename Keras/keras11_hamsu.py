#1. 데이터
import numpy as np

# x = np.array(range(1, 101))
# y = np.array(range(1, 101))
x = np.array([range(100), range(311, 411), range(100)])
y = np.array([range(501, 601), range(711, 811), range(100)])
# x = np.array([range(100), range(311, 411)]).T
# y = np.array([range(501, 601), range(711, 811)]).reshape(100,2)

x = np.transpose(x)
y = np.transpose(y)

print(x.shape)
print(y.shape)

#train 60프로 test 40프로
from sklearn.model_selection import train_test_split #비율 맞춰 잘라주는 함수 #열 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state = 66, test_size = 0.4
)

#test와 val이 절반씩 나눠가짐
x_test, x_val, y_test, y_val = train_test_split(
    x_test, y_test, random_state = 66, test_size = 0.5
)
#결과적으로 트레인60 테스트20 발20
# print(x_test.shape)


#2. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape=(3, ))
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(3)(dense1)
dense3 = Dense(4)(dense2)
output1 = Dense(3)(dense3)

model = Model(input=input1, output=output1)
model.summary()

# model.add(Dense(5, input_shape = (3, ), activation = 'relu'))
# model.add(Dense(3))
# model.add(Dense(4))
# model.add(Dense(3))




#3. 훈련
# model.compile(loss = 'mse', optimizer = 'adam', metrics=['accuracy'] )
model.compile(loss = 'mse', optimizer = 'adam', metrics=['mse'] )

# model.fit(x, y, epochs=610, batch_size=3)
model.fit(x_train, y_train, epochs=100, batch_size=1,
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