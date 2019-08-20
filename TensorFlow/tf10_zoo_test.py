import tensorflow as tf
import numpy as np
from keras.utils import np_utils

tf.set_random_seed(777)

xy = np.loadtxt('./data/data-04-zoo.csv', delimiter=',', dtype=np.float32)
print(xy.shape)   # (119, 17)

x_data = xy[: , 0:-1]
y_data = xy[: , [-1]]
y_data = np_utils.to_categorical(y_data)
print(x_data.shape, y_data.shape)   # (101, 16) (101, 7)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 16])
Y = tf.placeholder(tf.float32, shape=[None, 7])

W = tf.Variable(tf.random_normal([16, 7]), name='weight')
b = tf.Variable(tf.random_normal([7]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(-tf.matmul(X, W)))
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# cost/loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                                 labels=tf.stop_gradient([]))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Accuracy computatiom
# True if hypothesis > 0.5 else False
predicted = tf.argmax(hypothesis, 1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report
    _, c, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict={X: x_data, Y: y_data})
    print("\nCorrect (Y):\n", np.argmax(c, axis=1), "\nAccuracy: ", a)
    # print("\nHypothesis:\n", h, "\nCorrect (Y):\n", c, "\nAccuracy: ", a)