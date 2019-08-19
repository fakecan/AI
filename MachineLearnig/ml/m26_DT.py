# RandomSearch, K-fold, CV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LinearRegression

# 기온 데이터 읽어들이기
df = pd.read_csv('./data/tem10y.csv', encoding="utf-8")

# 데이터를 학습 전용과 테스트 전용으로 분리하기
train_year = (df["연"] <= 2015)
test_year = (df["연"] >= 2016)

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

x_train, y_train = make_data(df[train_year])
x_test, y_test = make_data(df[test_year])

# parameters = [{
#     'criterion' : ['mse', 'string', 'optional'],
#     'max_depth': [2, 4, 6, 8, 10, 12]
# }]


# def create_hyperparameters():
#     batches = [10,20,30,40,50]
#     optimizers = ['rmsprop', 'adam', 'adadelta']
#     dropout = np.linspace(0.1, 0.5, 5)
#     return{'batch_size':batches, 'optimizer':optimizers, 'keep_prob':dropout}

def create_hyperparameters():
    criterions = ['mse', 'friedman_mse', 'mae']
    splitters = ['random', 'best']
    max_depths = [2, 4, 6, 8, 10, 12, 14, 16]
    min_samples_splits = [2, 3, 4, 5, 6]
    min_samples_leafs = [1, 2, 3, 4, 5]
    min_weight_fraction_leafs = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    max_featuress = ['auto', 'sqrt', 'log2', 'None']
    return{'criterion': criterions, 'splitter': splitters, 'max_depth': max_depths,
           'min_samples_split': min_samples_splits, 'min_samples_leaf': min_samples_leafs,
           'min_weight_fraction_leaf': min_weight_fraction_leafs #'max_features': max_featuress
             
           }

hyperparameters = create_hyperparameters()

kfold_cv = KFold(n_splits=5, shuffle=True)

clf = RandomizedSearchCV(DecisionTreeRegressor(),
                         hyperparameters,
                         n_iter=100,
                         n_jobs=-1,
                         verbose=1,
                         cv=kfold_cv)

clf.fit(x_train, y_train)

print('최적의 매개 변수 =', clf.best_estimator_)

y_predict = clf.predict(x_test)
# print('최종 정답률 = ', accuracy_score(y_test, y_predict))
last_score = clf.score(x_test, y_test)
print('최종 정답률 = ', last_score)

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

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)