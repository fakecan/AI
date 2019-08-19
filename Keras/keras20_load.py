#DNN 끝(가장 노말함)
#1. 데이터
import numpy as np

# x = np.array(range(1, 101))
# y = np.array(range(1, 101))
x = np.array([range(1000), range(3110, 4110), range(1000)])
y = np.array([range(5010, 6010)])
# x = np.array([range(100), range(311, 411)]).T
# y = np.array([range(501, 601), range(711, 811)]).reshape(100,2)
# print(x)

print(x.shape)
print(y.shape)

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


# #2. 모델 구성
from keras.models import load_model
model = load_model("savetest01.h5")

import keras
td_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, #import 해준 것임 고로 
                                      write_graph=True, write_images=True)



#3. 훈련
# model.compile(loss = 'mse', optimizer = 'adam', metrics=['accuracy'] )
model.compile(loss = 'mse', optimizer = 'adam', metrics=['acc'] )

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='acc', patience=300, mode='auto') #monitor loss acc
hist = model.fit(x, y, epochs=3000, batch_size=512, verbose=1, #monitor = 'val_loss' 'val_acc'
                 validation_data=(x_val, y_val), callbacks=[early_stopping, td_hist])


#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=8)
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

print("loss : ", loss) #0.001 이하