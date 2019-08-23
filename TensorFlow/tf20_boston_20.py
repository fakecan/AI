#   회귀
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

# input, output dimention
input_dim, output_dim = x_train.shape[-1], y_train.shape[-1]
print(input_dim, output_dim)

# input Layer
X = tf.placeholder(tf.float32, [None, input_dim])
Y = tf.placeholder(tf.float32, [None, output_dim])

L1 = tf.layers.dense(X, 16, activation=tf.nn.relu)
L1 = tf.layers.dropout(L1, 0.3)
L2 = tf.layers.dense(L1, 32, activation=tf.nn.relu)
L2 = tf.layers.dropout(L2, 0.3)
L3 = tf.layers.dense(L2, 18, activation=tf.nn.relu)
hypothesis = tf.layers.dense(L3, 1, activation=tf.nn.relu)

cost = tf.reduce_mean(tf.square(hypothesis - Y))
# train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)
train = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.5).minimize(cost)

# Launch the graph in Sesstion
from sklearn.metrics import r2_score, mean_squared_error

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(30001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_train, Y:y_train} )
        if step % 100 == 0: print('>>',step, 'cost:',cost_val)
    
    h = sess.run(hypothesis, feed_dict={X: x_test})
    r2Score = r2_score(y_test, h) # 높을수록 좋음.
    rmseScore = np.sqrt(mean_squared_error(y_test, h)) # 낮을 수록 좋다.
    print('R2:', r2Score, 'RMSE:', rmseScore)

# 100000 cost: 9.26363
# R2: 0.9263859516153254 RMSE: 2.480499625655286