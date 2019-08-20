import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# X, Y, W, b, hypothesis, cost, train
# sigmoid
# predict, accuracy

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.Variable(tf.random_normal([2, 100]), name='weight1')
b1 = tf.Variable(tf.random_normal([100]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

'''
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
W2 = tf.Variable(tf.random_normal([100, 50]), name='weight2')
b2 = tf.Variable(tf.random_normal([50]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([50, 80]), name='weight3')
b3 = tf.Variable(tf.random_normal([80]), name='bias3')
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random_normal([80, 4000]), name='weight4')
b4 = tf.Variable(tf.random_normal([4000]), name='bias4')
layer4 = tf.sigmoid(tf.matmul(layer3, W4) + b4)

W5 = tf.Variable(tf.random_normal([4000, 70]), name='weight5')
b5 = tf.Variable(tf.random_normal([70]), name='bias5')
layer5 = tf.sigmoid(tf.matmul(layer4, W5) + b5)

W6 = tf.Variable(tf.random_normal([70, 10500]), name='weight6')
b6 = tf.Variable(tf.random_normal([10500]), name='bias6')
layer6 = tf.sigmoid(tf.matmul(layer5, W6) + b6)

W7 = tf.Variable(tf.random_normal([10500, 20]), name='weight7')
b7 = tf.Variable(tf.random_normal([20]), name='bias7')
layer7 = tf.sigmoid(tf.matmul(layer6, W7) + b7)

W8 = tf.Variable(tf.random_normal([20, 300000]), name='weight8')
b8 = tf.Variable(tf.random_normal([300000]), name='bias8')
layer8 = tf.sigmoid(tf.matmul(layer7, W8) + b8)

W9 = tf.Variable(tf.random_normal([300000, 150]), name='weight9')
b9 = tf.Variable(tf.random_normal([150]), name='bias9')
layer9 = tf.sigmoid(tf.matmul(layer8, W9) + b9)

W10 = tf.Variable(tf.random_normal([150, 250]), name='weight10')
b10 = tf.Variable(tf.random_normal([250]), name='bias10')
layer10 = tf.sigmoid(tf.matmul(layer9, W10) + b10)

W11 = tf.Variable(tf.random_normal([250, 60]), name='weight11')
b11 = tf.Variable(tf.random_normal([60]), name='bias11')
layer11 = tf.sigmoid(tf.matmul(layer10, W11) + b11)

W12 = tf.Variable(tf.random_normal([60, 1]), name='weight12')
b12 = tf.Variable(tf.random_normal([1]), name='bias12')
hypothesis = tf.sigmoid(tf.matmul(layer11, W12) + b12)
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
'''

W2 = tf.Variable(tf.random_normal([100, 1]), name='weight2')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')
hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        _, cost_val, w_val = sess.run([train, cost, W2], feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            print(step, cost_val, w_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis:\n", h, "\nCorrect (Y):\n", c, "\nAccuracy: ", a)