from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import keras

# 데이터 읽어 들이기
wine = pd.read_csv("./data/winequality-white.csv", sep=";", encoding="utf-8")
# print(wine.shape)

y = wine["quality"]
label = LabelEncoder()
# label2 = OneHotEncoder()
label.fit(y)
y = label.transform(y)

y = np_utils.to_categorical(y, 7)
print(y)
x = wine.drop("quality", axis=1)
# dataset = np.array(wine)


# x = dataset[:, :-1] 
# y = dataset[:, -1]
# print(x.shape)
# print(y.shape)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, train_size=0.8, shuffle=True
)

# x_train = np.reshape(x_train, (x_train.shape[0], 11, 1))
# print(x_train.shape)
# x_test = np.reshape(x_test, (x_test.shape[0], 11, 1))
# print(x_test.shape)

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
# print(x_train_scaled)
x_test_scaled = scaler.transform(x_test)
# print(x_train_scaled)

# 모델의 설정
model = Sequential()
model.add(Dense(128, input_dim=11, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))


# 모델 컴파일
model.compile(loss='categorical_crossentropy', # 이진분류모델
                optimizer='adam',
                metrics=['accuracy'])

# 모델 실행
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
hist = model.fit(x_train, y_train, epochs=512, batch_size=8, callbacks=[early_stopping])

# 결과 출력
# print("\n Accuracy: %.4f" % (model.evaluate(x_test, y_test)[1]))
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("acc : ", acc)
print("loss : ", loss)
