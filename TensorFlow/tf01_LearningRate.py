# 1. 데이터
import numpy as np
x = np.array([1, 2, 3, 4])
y = np.array([1, 2, 3, 4])

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 1, activation = 'relu')) #input_dim: 1개 밀어넣음
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

# 3. 훈련
from keras.optimizers import Adam, SGD, rmsprop, Adagrad, Adadelta, Adamax, Nadam
optimizer = Nadam(lr=0.01)
# Adam=0.001, rmsprop=0.001, Adagrad=0.01, Adadelta=1, Admax=0.01, Nadam=0.01
# model.compile(loss = 'mse', optimizer = 'adam', metrics=['mse'] )
model.compile(loss = 'mse', optimizer = optimizer, metrics=['mse'] )
model.fit(x, y, epochs=100, batch_size=1)

# 4. 평가 예측
mse, _ = model.evaluate(x, y, batch_size=1)
print("mse : ", mse)
ped1 = model.predict([1.5, 2.5, 3.5])
print(ped1)