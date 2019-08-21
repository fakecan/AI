import pandas as pd
from sklearn.model_selection import KFold, train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
import warnings
from sklearn.utils.testing import all_estimators # all_estimators: 모두 평가
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# 붓꽃 데이터 읽어들이기
iris_data = pd.read_csv('./data/iris2.csv', encoding="utf-8")

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:,"Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

# 학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8)

# 그리드 서치에서 사용할 매개변수 --- (*1)
parameters = [{
    "n_estimators":[100, 200, 300, 400, 500],
    "max_depth": [10, 20, 30, 40]
}]

# 그리드 서치 --- (*2)
kfold_cv = KFold(n_splits=5, shuffle=True)
clf = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold_cv)

clf.fit(x_train, y_train)
print("최적의 매개변수 = ", clf.best_estimator_)

# 최적의 매개변수로 평가하기 --- (*3)
y_pred = clf.predict(x_test)
print("최종 정답률 = ", accuracy_score(y_test, y_pred))
last_score = clf.score(x_test, y_test)
print("최종 정답률 = ", last_score)