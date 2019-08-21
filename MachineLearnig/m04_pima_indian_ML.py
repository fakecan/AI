from sklearn.svm import LinearSVC, SVC # Support Vector Machine, 선형회귀
from sklearn.metrics import accuracy_score # 정확도
from sklearn.neighbors import KNeighborsClassifier
import numpy
import tensorflow as tf

# seed 값 생성
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 로드
dataset = numpy.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",") # .: 현재 루트
X = dataset[:,0:8]
Y = dataset[:,8]

print(dataset.shape)

# 모델의 설정
# model = KNeighborsClassifier(n_neighbors=1)
model = SVC()

# 모델 실행
model.fit(X, Y)

#4. 평가 및 예측
y_predict = model.predict(X)
print("acc : ", accuracy_score(Y, y_predict))




'''
X_test = dataset[0:20,:]

# 모델의 설정
model = KNeighborsClassifier(n_neighbors=1)
# model = SVC()

# 모델 실행

model.fit(X, Y)

#4. 평가 및 예측
y_predict = model.predict(X_test)
print("acc : ", accuracy_score(Y, y_predict))
'''