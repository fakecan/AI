import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
import warnings
from sklearn.utils.testing import all_estimators # all_estimators: 모두 평가

warnings.filterwarnings('ignore')

# 붓꽃 데이터 읽어들이기
iris_data = pd.read_csv('./data/iris2.csv', encoding="utf-8")

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:,"Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

# classifier 알고리즘 모두 추출하기
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter="classifier")

# K-분할 크로스 밸리데이션 전용 객체
kfold_cv = KFold(n_splits=5, shuffle=True) # n_splits: 데이터를 5조각으로(4조각은 train)

for(name, algorithm) in allAlgorithms:
    # 각 알고리즘 객체 생성하기
    clf = algorithm()
    
    # score 메서드를 가진 클래스를 대상으로 하기
    if hasattr(clf, "score"):

        # 크로스 밸리데이션
        scores = cross_val_score(clf, x, y, cv=kfold_cv)
        print(name,"의 정답률 = ")
        print(scores)

