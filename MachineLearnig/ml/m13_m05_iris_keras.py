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

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("./data/iris.csv", encoding="UTF-8", names=['a','b','c','d','y'])
# print(iris_data)
print(iris_data.shape)
print(type(iris_data))

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, "y"]
#└ pandas의 series data는 내용을 보려면 values 이용
# ┗ y data들은 모두 문자열이므로 softmax로 분류가 불가능하므로 분류하기 위해서 문자열을 치환(oneHot Encoding)하여 표현해줘야한다
x = iris_data.loc[:, ["a",'b','c','d']]


# y2 = iris_data.iloc[:,4]
# x2 = iris_data.iloc[:36]

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
# lenc = LabelEncoder()
# y2 = lenc.fit_transform(y.values.reshape(-1,1)).toarray()
enc = OneHotEncoder()
y2 = enc.fit_transform(y.values.reshape(-1,1)).toarray()
# ┗ y의 data들은 정수값이 아니므로 keras.util.to_catagorical() 적용할 수 없다 -> OneHotEncoder 적용하게 되면 문자열을 bit표현으로 처리해준다

# print(y[0], " -- one hot enocding --> ", y2[0])
# print(y[50], " -- one hot enocding --> ", y2[50])
# print(y[100], " -- one hot enocding --> ", y2[100])


# 학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(x,y2,test_size=0.2, train_size=0.8, shuffle=True)
# └ Iris Data 전체적으로 섞어주기 위해서 shuffle=True 주어 섞이게 한다

def build_network(keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape=(4, ), name='input')
    x = Dense(128, activation='relu', name='hidden1')(inputs)
    x = Dropout(keep_prob)(x)
    x = Dense(128, activation='relu', name='hidden2')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(keep_prob)(x)

    prediction = Dense(3, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return model

def create_hyperparameters():
    batches = [1, 2, 4, 6, 8, 12, 16, 32, 64]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    return{'batch_size':batches, 'optimizer':optimizers, 'keep_prob':dropout}

from keras.wrappers.scikit_learn import KerasClassifier # 사이킷런과 호환하도록 함
model = KerasClassifier(build_fn=build_network, verbose=1)

hyperparameters = create_hyperparameters()

search = RandomizedSearchCV(estimator=model,
                        param_distributions=hyperparameters,
                        n_iter=10, n_jobs=2, cv=3, verbose=1)
                        # 작업이 10회 수행, 3겹 교차검증 사용

search.fit(x_train, y_train)

print(search.best_params_)
