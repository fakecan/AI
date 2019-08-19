from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)
print(train_targets.shape)
# print(test_data.shape)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
# scaler = StandardScaler()
scaler = MinMaxScaler()
scaler.fit(train_data)
scaler.fit(test_data)

train_data_scaled = scaler.transform(train_data)
test_data_scaled = scaler.transform(test_data)
# print(train_data_scaled)
# print(test_data_scaled)

# 데이터 표준화(전처리 작업)
# ##############################
# mean = train_data.mean(axis=0)
# train_data -= mean
# std = train_data.std(axis=0)
# train_data /= std

# test_data -= mean
# test_data /= std
# ##############################

from keras import models
from keras import layers

def build_model():
    #동일한 모델을 여러 번 생성할 것이므로 함수를 만들어 사용합니다
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                            input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae']) #mae 사용
    return model

#from sklearn.model_selection import StratifiedKFold

import numpy as np
k = 5 #4
num_val_samples = len(train_data) // k
# print(len(train_data))
# print(num_val_samples)

num_epochs = 50 #100
all_scores = []
for i in range(k):
    print('처리 중인 폴드 #', i)
    #검증 데이터 준비: k번째 분할
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    #훈련 데이터 준비: 다른 분할 전제
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
         axis=0)

    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
         axis=0)
    

    #케라스 모델 구성(컴파일 포함)
    model = build_model()
    #모델 훈련(verbose=0이므로 훈련 과정이 출력되지 않습니다)
    model.fit(partial_train_data, partial_train_targets,
                epochs=num_epochs, batch_size=1, verbose=0) #verbose=0
    #검증 세트로 모델 평가
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0) #mae 반환
    all_scores.append(val_mae)

print(all_scores)
print(np.mean(all_scores)) #평균값 출력

#StratifiedKFold 적용
#StandardScaler나 MinMaxScaler 선택
#score 1이하로