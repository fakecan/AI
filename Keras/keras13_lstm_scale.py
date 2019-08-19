from numpy import array #np를 임포트해서 numpy(np)를 표시하지 않아도 쌉가능
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],[6,7,8],[7,8,9],
            [8,9,10],[9,10,11],[10,11,12], [20,30,40],[30,40,50],[40,50,60]]) #x 10~12 #14뽑아내기 과제
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
# print(x)

#훈련할 데이터만 fit하고 val이나 test는 transform으로 적용만 하면 됨, x값으로 가능
#스탠다드스칼라 적용
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
print(x)

#민맥스
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x) #전환
# x = scaler.transform(x) #적용
# print(x)



print("x.shape : ", x.shape)
print("y.shape : ", y.shape) #y의 shape (4,)는 4가 결과값이라고 생각

# x = x.reshape((x.shape[0], x.shape[1], 1))
# print("x.shape : ", x.shape)

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
model.fit(x, y, epochs=2000, batch_size=3)

#4. 평가 및 예측
x_input = array([11,12,13])
x_input = x_input.reshape((1,3,1)) #1행 3열. 1,3, ?
yhat = model.predict(x_input)
print(yhat)

x_input = array([25,35,45])
x_input = x_input.reshape((1,3,1))
yhat = model.predict(x_input)
print(yhat)
'''