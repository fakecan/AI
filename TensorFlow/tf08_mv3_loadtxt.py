# Multi-variable linear regression 3
import tensorflow as tf
import numpy as np
tf.set_random_seed(777) # for reproducibility

# xy = np.loadtxt('D:/AI/TensorFlow/data/data-01-test-score.csv', delimiter=',', dtype=np.float32)
xy = np.loadtxt('./data/data-01-test-score.csv', delimiter=',', dtype=np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# Make sure the shape and data are OK
print(x_data, "\nx_data shape: ", x_data.shape)
print(y_data, "\ny_data shape: ", y_data.shape)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                    feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print("Step: ", step, ", Cost: ", cost_val, ", Prediction:\n", hy_val)