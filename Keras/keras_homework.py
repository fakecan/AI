#1. 데이터
import numpy as np
x_train = np.arange(1, 101, 1)
y_train = np.arange(501, 601, 1)
x_test = np.arange(1001, 1101, 1)
y_test = np.arange(1101, 1201, 1)
# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)


#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential() #순차적으로, 모델링

model.add(Dense(6, input_dim = 1, activation = 'relu')) #input_dim: 1개 밀어넣음
model.add(Dense(11))
model.add(Dense(21))
model.add(Dense(37))
model.add(Dense(21))
model.add(Dense(37))
model.add(Dense(16))
model.add(Dense(7))
model.add(Dense(1))


# model.summary()

#3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics=['accuracy'] )
# model.fit(x, y, epochs=610, batch_size=3)
model.fit(x_train, y_train, epochs=600)
#300일때 x값은 맞음381

#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("acc : ", acc)

yt_predict = model.predict(x_train) #y는 훈련시킨 값
print(yt_predict)

y_predict = model.predict(x_test) #x는 훈련시킨 값
print(y_predict)
