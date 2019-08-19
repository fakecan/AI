#mnist : 분류 모델

from keras.datasets import mnist #6만장에 대한 데이터
from keras.utils import np_utils 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy
import os
import tensorflow as tf


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# import matplotlib.pyplot as plt #이미지 시각화
# digit = X_train[5900]
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show() #plt에 있던 것 출력, jupyter notebook은 안써도 됨.

#데이터 불러오기
#스칼라: 데이터 한 개 #벡터: 연결된 데이터
#0과 1사이로 MinMaxscaler 한 것 /255
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255 #X_train.shape[0] 부분 6만 #17, 18라인 데이터 전처리
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
# print(Y_train.shape)
# print(Y_test.shape)
Y_train = np_utils.to_categorical(Y_train) #분류됨
Y_test = np_utils.to_categorical(Y_test)
# print(Y_train.shape) #(60000, 10) 뒤에 10은 데이터가 10. 3이면 0001 4면 00001 
# print(Y_test.shape)
#OneHot Incoding -- 카테고리컬이 만든

# print(X_train.shape)
# print(X_test.shape)

#컨볼루션 신경망의 설정
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten()) #데이터를 평평하게 편다.
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax')) #지정된 분류 모델 사용 시, 마지막은 무조건
                                           #softmax를 사용해야 한다.
                                            #10개 중에 하나 준다.(OneHot Incoding)

model.compile(loss='categorical_crossentropy', #분류 모델이라 사용한다.
              optimizer='adam',
              metrics=['accuracy'])

# model.summary()


# 모델 최적화 설정
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

#모델의 실행
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                    epochs=1, batch_size=2000, verbose=1, #epochs=30, batch_size=
                    callbacks=[early_stopping_callback]) #,checkpointer])
                   
#테스트 정확도 출력
print("\n Test Accuracy : %.4f" % (model.evaluate(X_test, Y_test)[1]))
#분류 모델값은 acc가 정확하다.

print(history.history.keys())