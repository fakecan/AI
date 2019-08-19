import numpy as np
import pandas as pd

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(X_train.shape)    # (60000, 28, 28)
# print(X_test.shape)     # (10000, 28, 28)
# print(Y_train.shape)    # (60000,)
# print(Y_test.shape)     # (10000,)
mnist_x = np.vstack((np.array(x_train), np.array(x_test)))
mnist_y = np.hstack((np.array(y_train), np.array(y_test)))
np.save("mnist_x.npy",mnist_x)
np.save("mnist_y.npy",mnist_y)
x = np.load("mnist_x.npy")
y = np.load("mnist_y.npy")
# mnist_train =
print(x.shape)  # (70000, 28, 28)
print(y.shape)  # (70000,)


from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(X_train.shape)    # (50000, 32, 32)
# print(X_test.shape)     # (10000, 32, 32)
# print(Y_train.shape)    # (50000, 1)
# print(Y_test.shape)     # (10000, 1)
cifar10_x = np.vstack((np.array(x_train), np.array(x_test)))
cifar10_y = np.vstack((np.array(y_train), np.array(y_test)))
np.save("cifar10_x.npy",cifar10_x)
np.save("cifar10_y.npy",cifar10_y)
x = np.load("cifar10_x.npy")
y = np.load("cifar10_y.npy")
# mnist_train =
print(x.shape)  # (60000, 32, 32, 3)
print(y.shape)  # (60000, 1)


from keras.datasets import boston_housing
(x_train, y_train),(x_test,y_test) = boston_housing.load_data()
boston_housing_x = np.vstack((np.array(x_train), np.array(x_test)))
boston_housing_y = np.hstack((np.array(y_train), np.array(y_test)))

np.save("boston_housing_x.npy",boston_housing_x)
np.save("boston_housing_y.npy",boston_housing_y)
x = np.load("boston_housing_x.npy")
y = np.load("boston_housing_y.npy")
# mnist_train =
print("boston_housing_x",x.shape)   # boston_housingx (506, 13)
print("boston_housing_y",y.shape)   # boston_housingy (506,)

# from skleran.datasets import load_boston

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
label = cancer.target.reshape(-1,1)
print("cancer_ori",cancer.data.shape)       # cancer_ori (569, 30)   
print("cancer_ori",cancer.target.shape)     # cancer_ori (569,) 
cancer_data = np.c_[cancer.data,label]

np.save("cancer_data.npy",cancer_data)
cancer_d = np.load("cancer_data.npy")
print("cancer",cancer_d.shape)              # cancer (569, 31)


iris_data = pd.read_csv("../data/iris2.csv", encoding="utf-8")


'''
from sklearn.datasets import load_boston
boston = load_boston()
# print(boston['data'].shape)     # (506, 13) # boston.data.shape
# print(boston['target'].shape)   # (506,)    # boston.target.shape
np.save("boston.npy", boston)

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
np.save("cancer.npy", cancer)
# print(cancer['data'].shape)     # (569, 30)
# print(cancer['target'].shape)   # (569,)


pima_indians_diabetes_data = np.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",")
np.save("pima-indians-diabetes.npy", pima_indians_diabetes_data)

iris_data = pd.read_csv("./data/iris.csv", encoding="UTF-8")
np.save("iris.npy", iris_data)

iris2_data = pd.read_csv("./data/iris2.csv", encoding="UTF-8")
np.save("iris2.npy", iris2_data)

winequality_white_data = pd.read_csv("./data/winequality-white.csv", encoding="UTF-8")
np.save("winequality-white.npy", winequality_white_data)

housing_data = np.loadtxt("./data/housing.csv", encoding="UTF-8")
np.save("housing.npy", housing_data)
'''