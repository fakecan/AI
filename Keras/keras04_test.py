#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])


#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential() #순차적으로, 모델링

model.add(Dense(6, input_dim = 1, activation = 'relu')) #input_dim: 1개 밀어넣음
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(9))
model.add(Dense(1))

# model.summary()


#3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics=['accuracy'] )
# model.fit(x, y, epochs=610, batch_size=3)
model.fit(x_train, y_train, epochs=200)


#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("acc : ", acc)

y_predict = model.predict(x_test) #x는 훈련시킨 값
print(y_predict)
