# m06_wine3.py를 keras로 리폼
# 96이상 나오게

# RP -> XG_boost -> keras 랜덤포레스트 먼저 써보고 안되면 케라스
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Dropout
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import keras

# 데이터 읽어 들이기
wine = pd.read_csv("./data/winequality-white.csv", sep=";", encoding="utf-8")

# 데이터를 레이블과 데이터로 분리하기
y = wine["quality"]
x = wine.drop("quality", axis=1)

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# y 레이블 변경하기
newlist = []
for v in list(y):
    if v <= 4:
        newlist += [0]
    elif v <= 7:
        newlist += [1]
    else:
        newlist += [2]
y = newlist

label = LabelEncoder() # label2 = OneHotEncoder()
label.fit(y)
y = label.transform(y)
y = np_utils.to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.4, train_size=0.6, shuffle=True
)

x_test, x_val, y_test, y_val = train_test_split(
    x_test, y_test, test_size = 0.5
)

# 모델의 설정
model = Sequential()
model.add(Dense(64, input_shape=(11, ), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

# 모델 컴파일
model.compile(loss='categorical_crossentropy', # 이진분류모델
                optimizer='adam',
                metrics=['accuracy'])

# 모델 실행
early_stopping = EarlyStopping(monitor='loss', patience=32, mode='auto')
hist = model.fit(x_train, y_train, epochs=256, batch_size=8,
                    callbacks=[early_stopping], validation_data=(x_val, y_val))

# 결과 출력
# print("\n Accuracy: %.4f" % (model.evaluate(x_test, y_test)[1]))
loss, acc = model.evaluate(x_test, y_test)#, batch_size=1)
print("acc : ", acc)
print("loss : ", loss)
