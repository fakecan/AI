#1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x2 = np.array([4,5,6])


#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential() #순차적으로, 모델링

model.add(Dense(5, input_dim = 1, activation = 'relu')) #input_dim: 1개 밀어넣음
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))
# epochs = 600

# model.add(Dense(30, input_dim = 1, activation = 'relu')) #input_dim: 1개 밀어넣음
# model.add(Dense(70))
# model.add(Dense(30))
# model.add(Dense(50))
# model.add(Dense(38))
# model.add(Dense(48))
# model.add(Dense(20))
# model.add(Dense(36))
# model.add(Dense(20))
# model.add(Dense(50))
# model.add(Dense(1))
# # epochs = 600


#3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics=['accuracy'] )
# model.fit(x, y, epochs=610, batch_size=3)
model.fit(x, y, epochs=100)


#4. 평가 예측
loss, acc = model.evaluate(x, y, batch_size=32)
print("acc : ", acc)

y_predict = model.predict(x2) #x는 훈련시킨 값
print(y_predict)