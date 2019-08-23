from keras.datasets import cifar10 # from tensorflow.keras.datasets.cifar10 import load_data
from keras.utils import np_utils
import tensorflow as tf
import random
import keras

tf.set_random_seed(777)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)
# print(y_train.shape, y_test.shape)  # (50000, 1) (10000, 1)

# hyper parameters
learning_rate = 0.001
training_epochs = 100

# ■■■■■■■■■■ categorical ■■■■■■■■■■
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# input place holders
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
X_img = tf.reshape(X, [-1, 32, 32, 3])  # img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 10])

#
L1 = tf.layers.conv2d(X_img, 32, [3, 3], activation=tf.nn.relu)
L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2])
# L1 = tf.layers.dropout(L1, 0.3)
print('L1: ', L1)   # shape=(?, 15, 15, 32)

L2 = tf.layers.conv2d(L1, 64, [3, 3], activation=tf.nn.relu)
L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2])
# L2 = tf.layers.dropout(L1, 0.3)
print('L2: ', L2)   # shape=(?, 15, 15, 32)

L3 = tf.layers.conv2d(L2, 32, [3, 3], activation=tf.nn.relu)
L3 = tf.layers.max_pooling2d(L3, [2, 2], [2, 2])
# L3 = tf.layers.dropout(L3, 0.3)
print('L3: ', L3)   # shape=(?, 6, 6, 32)

L4 = tf.contrib.layers.flatten(L3)
L4 = tf.layers.dense(L4, 16, activation=tf.nn.relu)
# L4 = tf.layers.dropout(L4, 0.3)

# L5 = tf.contrib.layers.flatten(L4)
# L5 = tf.layers.dense(L5, 16, activation=tf.nn.relu)
# L5 = tf.layers.dropout(L5, 0.3)
logits = tf.layers.dense(L4, 10, activation=None)

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Train')
    for step in range(training_epochs):
        avg_cost = 0

        for i in range(1000):
            batch_xs, batch_ys = x_train[i*50:i*50+50], y_train[i*50:i*50+50]
            cost_val, _ = sess.run([cost, train], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += cost_val/ 1000
        print('Epoch>>', step, ' Cost>>', avg_cost)

    print('Learning Finished!')

    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))
