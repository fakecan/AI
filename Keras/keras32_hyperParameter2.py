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
# X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255 #X_train.shape[0] 부분 6만 #17, 18라인 데이터 전처리
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
X_train = X_train.reshape(X_train.shape[0], 28*28).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28*28).astype('float32') / 255
print(Y_train.shape)
print(Y_test.shape)
Y_train = np_utils.to_categorical(Y_train) #분류됨
Y_test = np_utils.to_categorical(Y_test)
print(Y_train.shape) #(60000, 10) 뒤에 10은 데이터가 10. 3이면 0001 4면 00001 
print(Y_test.shape)
#OneHot Incoding -- 카테고리컬이 만든
# print(X_train.shape)
# print(X_test.shape)


from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input
import numpy as np

def build_network(keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape=(28*28, ), name='input')
    dense1 = Dense(512, activation='relu', name='hidden1')(inputs)
    dense1= Dropout(keep_prob)(dense1)
    dense2 = Dense(256, activation='relu', name='hidden2')(dense1)
    dense2 = Dropout(keep_prob)(dense2)
    dense3 = Dense(128, activation='relu', name='hidden3')(dense2) #얘만 돌아감
    dense3 = Dropout(keep_prob)(dense3)
    prediction = Dense(10, activation='softmax', name='output')(dense3)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return model

def create_hyperparameters():
    batches = [10,20,30,40,50]
    optimizer = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    return{"batch_size":batches, "optimizer":optimizer, "keep_prob":dropout}

from keras.wrappers.scikit_learn import KerasClassifier #사이킷런과 호환하도록 한다 #분류(mnist같은)
#from keras.wrappers.scikit_learn import KerasRegressor #사이킷런과 호환하도록 한다
model = KerasClassifier(build_fn=build_network, verbose=1) #verbose=0 #build_fn: 모델을 땡겨오겠다

hyperparameters = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(estimator=model,
                            param_distributions=hyperparameters,
                            n_iter=10, n_jobs=1, cv=3, verbose=1) #10x3=30
                            #작업이 10회 수행, 3겹 교차검증 사용 

# search.fit(data["X_train"], data["y_train"])
search.fit(X_train, Y_train)

print(search.best_params_) #create_hyperparameters() 각 변수의 최상의 변수 값

#과제 CNN 모델로 바꾸자