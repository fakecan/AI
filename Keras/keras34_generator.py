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

X_train = X_train[:300]
X_test = X_test
Y_train = Y_train[:300]
Y_test = Y_test

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255 #X_train.shape[0] 부분 6만 #17, 18라인 데이터 전처리
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

#컨볼루션 신경망의 설정
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten()) #데이터를 평평하게 편다.
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax')) #지정된 분류 모델 사용 시, 마지막은 무조건
                                           #softmax를 사용해야 한다.
                                            #10개 중에 하나 준다.(OneHot Incoding)

model.compile(loss='categorical_crossentropy', #분류 모델이라 사용한다.
              optimizer='adam',
              metrics=['accuracy'])

# model.summary()


# 모델 최적화 설정
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=50)

#모델의 실행
from keras.preprocessing.image import ImageDataGenerator
data_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.02,
    height_shift_range=0.02,
    horizontal_flip=True
)

model.fit_generator(data_generator.flow(X_train, Y_train, batch_size=32),
                    # steps_per_epoch=len(X_train) * 20 // 32, #몇배로 증폭시킬것이냐
                    steps_per_epoch = 20,
                    epochs=200,
                    validation_data=(X_test, Y_test),
                    verbose=1, #callbacks=callbacks
                    callbacks=[early_stopping_callback]
                    )

# history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
#                     epochs=30, batch_size=10, verbose=1, #epochs=30
#                     callbacks=[early_stopping_callback]) #,checkpointer])
                   
#테스트 정확도 출력
print("\n Test Accuracy : %.4f" % (model.evaluate(X_test, Y_test)[1]))
#분류 모델값은 acc가 정확하다.




#25번에 300개 잘라놓은 것을 긁어다가 이미지를 6만개로 만들기
#증폭시켜서 CNN 돌리기