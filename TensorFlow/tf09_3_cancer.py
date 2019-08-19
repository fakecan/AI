# Logistic Regression Classifier
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_breast_cancer
tf.set_random_seed(777) # for reproducibility

cancer = load_breast_cancer()

# x_data = cancer.data
# y_data = cancer.target
x_data = cancer['data']
y_data = cancer['target']
print(x_data.shape, y_data.shape) #(569, 30) (569,)
y_data = y_data.reshape(-1, 1)
print(x_data.shape, y_data.shape)

# placeholdes for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 30])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([30, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(-tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computatiom
# True if hypothesis > 0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis:\n", h, "\nCorrect (Y):\n", c, "\nAccuracy: ", a)
