# RandomSearch 적용
import pandas as pd
from sklearn.model_selection import train_test_split 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, LSTM
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer() # 분류
# print(cancer.DESCR)

# df = pd.DataFrame(np.c_[cancer['data'], cancer['target']],
#                   columns= np.append(cancer['feature_names'], ['target']))
# df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
# df['target'] = cancer.target
# print(df.head())

x = cancer['data']
y = cancer['target']
print(x.shape) #(569, 30)
print(y.shape) #(569,)


# 학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, train_size=0.8)
# └ Iris Data 전체적으로 섞어주기 위해서 shuffle=True 주어 섞이게 한다

def build_network(keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape=(30, ), name='input')
    x = Dense(256, activation='relu', name='hidden1')(inputs)
    x = Dropout(keep_prob)(x)
    x = Dense(128, activation='relu', name='hidden2')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(256, activation='relu', name='hidden3')(x)
    x = Dropout(keep_prob)(x)

    x = Dense(64, activation='relu', name='hidden4')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(32, activation='relu', name='hidden5')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(16, activation='relu', name='hidden6')(x)
    x = Dropout(keep_prob)(x)
    # x = Dense(64, activation='relu', name='hidden3')(x)
    # x = Dropout(keep_prob)(x)

    prediction = Dense(1, activation='sigmoid', name='output')(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                    metrics=['accuracy'])
    return model

def create_hyperparameters():
    batches = [1, 2, 4, 6, 8, 12]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0, 0.1, 2)
    return{'batch_size':batches, 'optimizer':optimizers, 'keep_prob':dropout}

from keras.wrappers.scikit_learn import KerasClassifier # 사이킷런과 호환하도록 함
model = KerasClassifier(build_fn=build_network, verbose=1)

hyperparameters = create_hyperparameters()

search = RandomizedSearchCV(estimator=model,
                        param_distributions=hyperparameters,
                        n_iter=10, n_jobs=10, cv=3, verbose=1)
                        # 작업이 10회 수행, 3겹 교차검증 사용

search.fit(x_train, y_train)

print(search.best_params_)
