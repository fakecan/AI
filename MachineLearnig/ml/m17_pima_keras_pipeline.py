# pima-indians-diabets.csv를 파이프라인으로 처리한다
# 최적의 파라미터를 구한 뒤 모델링해서 acc를 확인한다

# ctrl shift space

# RandomSearch 적용 #이진 분류 모델
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, LSTM
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
# import warnings
# warnings.filterwarnings('ignore')

# seed 값 생성
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 로드
dataset = np.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",") # .: 현재 루트
x = dataset[:,0:8]
y = dataset[:,8]

print(dataset.shape) # (768, 9)
print(x.shape) # (768, 8)
print(y.shape) # (768,)

# 학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, train_size=0.8, shuffle=True)
# └ Iris Data 전체적으로 섞어주기 위해서 shuffle=True 주어 섞이게 한다

def build_network(keep_prob=0.2, optimizer='adam'):
    inputs = Input(shape=(8, ), name='input')
    x = Dense(128, activation='relu', name='hidden1')(inputs)
    x = Dropout(keep_prob)(x)
    x = Dense(64, activation='relu', name='hidden2')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(96, activation='relu', name='hidden3')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(48, activation='relu', name='hidden4')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(32, activation='relu', name='hidden5')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(16, activation='relu', name='hidden6')(x)
    x = Dropout(keep_prob)(x)

    prediction = Dense(1, activation='sigmoid', name='output')(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                    metrics=['accuracy', 'mse', 'mae'])

    return model

def create_hyperparameters():
    batches = [2, 4, 6, 8, 10]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0, 0.3, 6)
    return{ 'kerasclassifier__batch_size':batches,
            'kerasclassifier__optimizer':optimizers,
            'kerasclassifier__keep_prob':dropout    }

from keras.wrappers.scikit_learn import KerasClassifier # 사이킷런과 호환하도록 함
model = KerasClassifier(build_fn=build_network, verbose=1)
hyperparameters = create_hyperparameters()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
# pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])

pipe = Pipeline([("scaler", MinMaxScaler()), ('kerasclassifier', model)])
search = RandomizedSearchCV(estimator=pipe,
                        param_distributions=hyperparameters,
                        n_iter=100, n_jobs=-1, cv=5, verbose=1)
                        # 작업이 10회 수행, 3겹 교차검증 사용
search.fit(x_train, y_train)

y_predict = search.predict(x_test)
last_score = search.score(x_test, y_test)

print("best_params_ : ", search.best_params_)
print("best_score_ : ", search.best_score_)
print("score : ", search.score(x_test,y_test))
print("acc : ", accuracy_score(y_test, y_predict))
print("최종 정답률 = ", last_score)
# print('predict: \n', search.predict(x_test))

# RMSE solution
from sklearn.metrics import mean_squared_error #레거시한 머신 러닝 중 하나
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# RMAE solution
from sklearn.metrics import mean_absolute_error
def RMAE(y_test, y_predict):
    return np.sqrt(mean_absolute_error(y_test, y_predict))
print("RMAE : ", RMAE(y_test, y_predict))

