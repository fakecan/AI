#   이진 분류
from sklearn.datasets import load_breast_cancer
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

tf.set_random_seed(777)

# hyper parameters
learning_rate = 0.001
training_epochs = 100

dataset = load_breast_cancer()
x_data, y_data = dataset.data, dataset.target
# print(x_data.shape, y_data.shape) #(569, 30) (569,)

y_data = y_data.reshape(-1, 1)
# print(x_data.shape, y_data.shape) #(569, 30) (569, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size = 0.2, random_state = 66
)
print(x_train.shape, x_test.shape)  # (455, 30) (114, 30)
print(y_train.shape, y_test.shape)  # (455, 1) (114, 1)


# ■■■■■■■■■■■■■■■■■■■■ Data Preprocessing ■■■■■■■■■■■■■■■■■■■■
def data_Preprocessing(train, test):
    for scaler in [MinMaxScaler()]:
        scaler.fit(train)
        train = scaler.transform(train)
        test = scaler.transform(test)
    return train, test

x_train, x_test = data_Preprocessing(x_train, x_test)

# input place holders
X = tf.placeholder(tf.float32, [None, 30])
Y = tf.placeholder(tf.float32, [None, 1])

L1 = tf.layers.dense(X, 16, activation=tf.nn.relu)
# L1 = tf.layers.dropout(L1, 0.3)
L2 = tf.layers.dense(L1, 32, activation=tf.nn.relu)
# L3 = tf.layers.dense(L2, 8, activation=tf.nn.relu)
logits = tf.layers.dense(L2, 1, activation=tf.nn.sigmoid)

# define cost/loss & optimizer
cost = -tf.reduce_mean(Y * tf.log(logits) + (1 - Y) * tf.log(1 - logits))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

predicted = tf.cast(logits > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = tf.float32))

# fit, accuracy
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(3001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_train, Y: y_train})
        if step % 10 == 0:
            print('Epoch>>', step, 'Cost>>', cost_val)
    
    # Predict, Accuracy
    h, c, a = sess.run( [logits, predicted, accuracy],
                        feed_dict={X: x_test, Y: y_test})
    print('Accuracy: ', a)
    