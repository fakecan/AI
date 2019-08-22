import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Dataset
from sklearn.datasets import load_boston
boston_dataset = load_boston()

data, target = boston_dataset.data, np.array(boston_dataset.target)
target = target.reshape(len(target), 1)
print(data.shape, target.shape) # (506, 13) (506, 1) 회귀

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.2, random_state = 66)
print(x_train.shape, y_train.shape) # (404, 13) (404, 1)
print(x_test.shape, y_test.shape) # (102, 13) (102, 1)

# Data Preprocessing
def data_prep(train, test):
    for scaler in [MinMaxScaler()]:
        scaler.fit(train)
        train = scaler.transform(train)
        test = scaler.transform(test)
    return train, test

x_train, x_test = data_prep(x_train, x_test)

#Tensorflow
tf.set_random_seed(777)

#Dense Class
class Dense:
    
    node = 1
    W = None
    b = None
    Y = None

    def __init__(self, X, node, initialize = None, w_name = 'w',input_dim = 0):
        self.node = node
        # X가 Dense일 경우
        if type(X) == type(self):
            input_dim = X.node
            tensor = X.Y
        else: # X가 Tensor일 경우..
            tensor = X

        # get_varialve 사용 여부
        if initialize == None:
            self.W = tf.Variable(tf.random_normal([input_dim, node]))
            self.b = tf.Variable(tf.random_normal([node]))
        else:
            self.W = tf.get_variable(w_name, shape = [input_dim, node], initializer = initialize)
            self.b = tf.Variable(tf.random_normal([node]))
        
        # Logic
        self.Y = tf.matmul(tensor, self.W) + self.b

    def relu(self):
        Y = self.Y
        self.Y = tf.nn.relu(Y)
        return self

    def softmax(self):
        Y = self.Y
        self.Y = tf.nn.softmax(Y)
        return self

    def sigmoid(self):
        Y = self.Y
        self.Y = tf.nn.sigmoid(Y)
        return self
    
    def dropout(self, keep_prob = 0.2):
        Y = self.Y
        self.Y = tf.nn.dropout(Y, keep_prob)
        return self

#Tensorflow
tf.set_random_seed(777)

#Dense Class
class Dense:
    
    node = 1
    W = None
    b = None
    Y = None

    def __init__(self, X, node, initialize = None, w_name = 'w',input_dim = 0):
        self.node = node
        # X가 Dense일 경우
        if type(X) == type(self):
            input_dim = X.node
            tensor = X.Y
        else: # X가 Tensor일 경우..
            tensor = X

        # get_varialve 사용 여부
        if initialize == None:
            self.W = tf.Variable(tf.random_normal([input_dim, node]))
            self.b = tf.Variable(tf.random_normal([node]))
        else:
            self.W = tf.get_variable(w_name, shape = [input_dim, node], initializer = initialize)
            self.b = tf.Variable(tf.random_normal([node]))
        
        # Logic
        self.Y = tf.matmul(tensor, self.W) + self.b

    def relu(self):
        Y = self.Y
        self.Y = tf.nn.relu(Y)
        return self

    def softmax(self):
        Y = self.Y
        self.Y = tf.nn.softmax(Y)
        return self

    def sigmoid(self):
        Y = self.Y
        self.Y = tf.nn.sigmoid(Y)
        return self
    
    def dropout(self, keep_prob = 0.2):
        Y = self.Y
        self.Y = tf.nn.dropout(Y, keep_prob)
        return self

# input, output dimention
input_dim, output_dim = x_train.shape[-1], y_train.shape[-1]
print(input_dim, output_dim)

# input Layer
X = tf.placeholder(tf.float32, [None, input_dim])
Y = tf.placeholder(tf.float32, [None, output_dim])

# Hidden Layer
initialize = tf.contrib.layers.xavier_initializer()

hLayer = Dense(X, 64, input_dim = input_dim)
hLayer = hLayer.dropout(0.5)
hLayer = Dense(hLayer.relu(), 64, initialize, 'w1')
hLayer = hLayer.dropout(0.5)
hLayer = Dense(hLayer.relu(), 64, initialize, 'w2')
hLayer = hLayer.dropout(0.5)
hLayer = Dense(hLayer.relu(), output_dim, initialize, 'w3')
hLayer = hLayer.relu()
hypothesis = hLayer.Y

W = hLayer.W
b = hLayer.b

cost = tf.reduce_mean(tf.square(hypothesis - Y))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

# Launch the graph in Sesstion
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_train, Y: y_train})
    if step % 40 == 0 : print(step, 'Cost', cost_val)

r2Score = r2_score(y_test, hy_val)
rmseScore = np.sqrt(mean_squared_error(y_test, hy_val))
print('R2:',r2Score, 'RMSE:', rmseScore)