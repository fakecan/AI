from keras.datasets import cifar10 # from tensorflow.keras.datasets.cifar10 import load_data
from keras.utils import np_utils
import tensorflow as tf
import random

tf.set_random_seed(777)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print('X_train shape : ', X_train.shape) #(50000, 32, 32, 3)
# print('X_test shape : ', X_test.shape) # (10000, 32, 32, 3)

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# ■■■■■■■■■■ categorical ■■■■■■■■■■
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# input place holders
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
X_img = tf.reshape(X, [-1, 32, 32, 3])  # img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                      strides=[1, 2, 2, 1], padding='SAME')
# print("L1: ", L1)   # shape=(?, 16, 16, 32)

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01)) # W1의 32
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                      strides=[1, 2, 2, 1], padding='SAME')
# print("L2: ", L2)   # shape=(?, 8, 8, 64)

W3 = tf.Variable(tf.random_normal([3, 3, 64, 32], stddev=0.01)) # W1의 32
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],
                      strides=[1, 2, 2, 1], padding='SAME')
# print("L3: ", L3)   # shape=(?, 4, 4, 32)
L3_flat = tf.reshape(L3, [-1, 4 * 4 * 32])

W4 = tf.get_variable("W4", shape=[4 * 4 * 32, 16],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([16]))
logits = tf.matmul(L3_flat, W4) + b4

W5 = tf.get_variable("W5", shape=[16, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(logits, W5) + b5

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
    for step in range(100):
        avg_cost = 0

        for i in range(500):
            batch_xs, batch_ys = x_train[i*100:i*100+100], y_train[i*100:i*100+100]
            cost_val, _ = sess.run([cost, train], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += cost_val/ 500
        print('Epoch>>', step, ' Cost>>', avg_cost)

    print('Learning Finished!')

    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))
