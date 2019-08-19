from numpy import array #np를 임포트해서 numpy(np)를 표시하지 않아도 쌉가능
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],[6,7,8],[7,8,9],
            [8,9,10],[9,10,11],[10,11,12], [20,30,40],[30,40,50],[40,50,60]]) #x 10~12 #14뽑아내기 과제
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70]) 

print("x.shape : ", x.shape)
print("y.shape : ", y.shape) #y의 shape (4,)는 4가 결과값이라고 생각

x = x.reshape((x.shape[0], x.shape[1], 1))
print("x.shape : ", x.shape)

'''
#2. 모델 구성
model = Sequential()
model.add(LSTM(310, activation='relu', input_shape=(3,1))) #  (4, 3, 1) 4행 3열 1개씩 자른다. 여기서는 4행 쳐내고 3열 1개 자름. 3열 1개씩
model.add(Dense(30))
model.add(Dense(300))
model.add(Dense(30))
model.add(Dense(300))
model.add(Dense(30))
model.add(Dense(1))

# model.summary()

#3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=1000, batch_size=3, verbose=0)
# verbose= 0생략, 1원래값, 2축약, 3더축약

#4. 평가 및 예측
x_input = array([25,35,45])
x_input = x_input.reshape((1,3,1)) #1행 3열. 1,3, ?
yhat = model.predict(x_input, verbose=1)
print(yhat)
#자세히 보려면 verbose = 0생략 1자세히 

# x_input = array([25,35,45])
# x_input = x_input.reshape((1,3,1))
# yhat = model.predict(x_input)
# print(yhat)
'''