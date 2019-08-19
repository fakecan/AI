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
import keras
td_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, #import 해준 것임 고로 
                                      write_graph=True, write_images=True)
# keras.callbacks.TensorBoard 와 TensorBoard는 같은말

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras import regularizers

model = Sequential() #순차적으로, 모델링

# model.add(Dense(5, input_dim = 1, activation = 'relu')) #input_dim: 1개 밀어넣음
model.add(Dense(10, input_shape = (3, ), activation = 'relu')) #input_dim: 1개 밀어넣음
# model.add(BatchNormalization())
# model.add(Dense(1000, kernel_regularizer=regularizers.l1(0.0001)))

model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1)) #y값이 2개
#레이어 5개 이상(입출포함), 노드는 5개이상
# model.summary()

model.save('savetest01.h5')