# RP -> XG_boost -> keras 랜덤포레스트 먼저 써보고 안되면 케라스
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier #랜덤으로 떨궈서 분류?
from sklearn.metrics import accuracy_score, classification_report

# 데이터 읽어 들이기
wine = pd.read_csv("./data/winequality-white.csv", sep=";", encoding="utf-8")

# 데이터를 레이블과 데이터로 분리하기
y = wine["quality"]
x = wine.drop("quality", axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2
)

# 학습하기
model = RandomForestClassifier(n_estimators=256,  n_jobs=-1,
        oob_score=True, random_state=0, max_features='auto')
model.fit(x_train, y_train)
aaa = model.score(x_test, y_test)

# 평가하기
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("정답률 = ", accuracy_score(y_test, y_pred))
print(aaa)

#케라스와 머신러닝의 차이
#케라스의 평가 부분을 머신러닝에서는 스코어로 쓴다