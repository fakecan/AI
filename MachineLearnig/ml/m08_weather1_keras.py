import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping


# 기온 데이터 읽어들이기
df = pd.read_csv('./data/tem10y.csv', encoding="utf-8")

# 데이터를 학습 전용과 테스트 전용으로 분리하기
train_year = (df["연"] <= 2015)
test_year = (df["연"] >= 2016)

# train_year = np.array(train_year)
# test_year = np.array(test_year)


interval = 6

# 과거 6일의 데이터를 기반으로 학습할 데이터 만들기
def make_data(data):
    x = [] # 학습 데이터
    y = [] # 결과
    temps = list(data["기온"])
    for i in range(len(temps)):
        if i < interval: continue
        y.append(temps[i])
        xa = []
        for p in range(interval):
            d = i + p - interval
            xa.append(temps[d])
        x.append(xa)
    return (x, y)

# train_year = np_utils.to_categorical(train_year)
# test_year = np_utils.to_categorical(test_year)

train_x, train_y = make_data(df[train_year])
test_x, test_y = make_data(df[test_year])

print(np.array(train_x).shape) # (3646, 6)
print(np.array(train_y).shape) # (3646,)
print(np.array(test_x).shape) # (4012, 6)
print(np.array(test_y).shape) # (4012,)


train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)
train_x = np.reshape(train_x, (train_x.shape[0], interval, 1))
test_x = np.reshape(test_x, (test_x.shape[0], interval, 1))

test_x, val_x, test_y, val_y = train_test_split(
    test_x, test_y, test_size = 0.5
)

# 모델의 설정
model = Sequential()
# model.add(LSTM(16, input_shape=(6, 1), activation='relu'))
# model.add(LSTM(8, return_sequences=True))

# model.add(LSTM(12, input_shape=(6, 1), return_sequences=True))
# model.add(LSTM(8))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='relu'))

# model.add(LSTM(40, input_shape=(interval, 1), return_sequences=True))
# model.add(LSTM(16, return_sequences=True))
# model.add(LSTM(10))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1, activation='relu')) #0.91
# 64 16 6 5 1

model.add(LSTM(34, input_shape=(interval, 1), return_sequences=True))
model.add(LSTM(20, return_sequences=True))
model.add(LSTM(14))
model.add(Dense(24, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))

# 모델 컴파일
model.compile(loss='mse',
                optimizer='adam', #adadelta',
                metrics=['mse'])

# 모델 실행
early_stopping = EarlyStopping(monitor='loss', patience=32, mode='auto')
hist = model.fit(train_x, train_y, epochs=300, batch_size=32,
                    callbacks=[early_stopping], validation_data=(val_x, val_y))

# 결과 출력
# print("\n Accuracy: %.4f" % (model.evaluate(x_test, y_test)[1]))
loss, acc = model.evaluate(test_x, test_y)#, batch_size=1)
print("acc : ", acc)
print("loss : ", loss)

pre_y = model.predict(test_x) #x는 훈련시킨 값
print(pre_y)

#RMSE 구하기
from sklearn.metrics import mean_squared_error #레거시한 머신 러닝 중 하나
def RMSE(test_y, pre_y):
    return np.sqrt(mean_squared_error(test_y, pre_y))
print("RMSE : ", RMSE(test_y, pre_y))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(test_y, pre_y)
print("R2 : ", r2_y_predict)

'''
# 결과를 그래프로 그리기
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(test_y, c='r')
plt.plot(pre_y, c='b')
plt.savefig('tenki-kion-lr.png')
plt.show()
'''
