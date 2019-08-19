#1. 데이터
import numpy as np

x1 = np.array([range(100), range(311, 411), range(100)])
y1 = np.array([range(501, 601), range(711, 811), range(100)])
x2 = np.array([range(100, 200), range(311, 411), range(100, 200)])
y2 = np.array([range(501, 601), range(711, 811), range(100)])

x1 = np.transpose(x1)
y1 = np.transpose(y1)
x2 = np.transpose(x2)
y2 = np.transpose(y2)

print(x1.shape)
print(x2.shape)
print(y1.shape)
print(y2.shape)


#train 60프로 test 40프로
from sklearn.model_selection import train_test_split #비율 맞춰 잘라주는 함수 #열 분할
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, random_state = 66, test_size = 0.4
)

#test와 val이 절반씩 나눠가짐
x1_val, x1_test, y1_val, y1_test = train_test_split(
    x1_test, y1_test, random_state = 66, test_size = 0.5
)

from sklearn.model_selection import train_test_split #비율 맞춰 잘라주는 함수 #열 분할
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, random_state = 66, test_size = 0.4
)

#test와 val이 절반씩 나눠가짐
x2_test, x2_val, y2_test, y2_val = train_test_split(
    x2_test, y2_test, random_state = 66, test_size = 0.5
)

print(x2_test.shape)
#결과적으로 트레인60 테스트20 발20

# print(x_test.shape)


#2. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input #함수형 모델링은 합칠 수가 있다.
# model = Sequential() #순차적으로, 모델링

input1 = Input(shape=(3, ))
dense1 = Dense(100, activation='relu')(input1) #100은 아웃풋 레이어
dense1_2 = Dense(30)(dense1) #두번째 레이어층
dense1_3 = Dense(7)(dense1_2)

input2 = Input(shape=(3, ))
dense2 = Dense(50, activation='relu')(input2)
dense2_2 = Dense(7)(dense2)

from keras.layers.merge import concatenate
merge1 = concatenate([dense1_3, dense2_2])

middle1 = Dense(10)(merge1)
middle2 = Dense(5)(middle1)
middle3 = Dense(7)(middle2)


############################################################ output 모델 구성

output1 = Dense(30)(middle3)
output1_2 = Dense(7)(output1)
output1_3 = Dense(3)(output1_2)

output2 = Dense(20)(middle3)
output2_2 = Dense(70)(output2)
output2_3 = Dense(3)(output2_2)

#5개의 모델

model = Model(inputs = [input1, input2],
              outputs = [output1_3, output2_3]
)

# model.summary()


#3. 훈련
# model.compile(loss = 'mse', optimizer = 'adam', metrics=['accuracy'] )
model.compile(loss = 'mse', optimizer = 'adam', metrics=['accuracy'] )

model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=50, batch_size=1,
          validation_data=([x1_val, x2_val], [y1_val, y2_val]))


#4. 평가 예측
# _, _, _, acc1, acc2 = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)
# print("acc1 : ", acc1)
# print("acc2 : ", acc2)

acc = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)
print("acc : ", acc)


# a = list(model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1))
# print(a)

y1_predict, y2_predict = model.predict([x1_test, x2_test]) #x는 훈련시킨 값
print("y1_predict : \n", y1_predict)
print("y2_predict : \n", y2_predict)

#RMSE 구하기
from sklearn.metrics import mean_squared_error #레거시한 머신 러닝 중 하나
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE : ", (RMSE1 + RMSE2)/2)

# R2 구하기
from sklearn.metrics import r2_score
r2_y1_predict = r2_score(y1_test, y1_predict)
r2_y2_predict = r2_score(y2_test, y2_predict)
print("R2_1 : ", r2_y1_predict)
print("R2_2 : ", r2_y2_predict)
print("R2 : ", (r2_y1_predict + r2_y2_predict)/2)

# #RMSE 구하기
# from sklearn.metrics import mean_squared_error #레거시한 머신 러닝 중 하나
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# print("RMSE : ", RMSE(y_test, y_predict))

# # R2 구하기
# from sklearn.metrics import r2_score
# r2_y_predict = r2_score(y_test, y_predict)
# print("R2 : ", r2_y_predict)