# # W1 = tf.get_variable("W1", shape=[?, ?],
# #                      initializer=tf.random_uniform_initializer())
# # b1 = tf.Variable(tf.random_normal([512]))
# # L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
# # L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# # tf.constant_initializer()
# # tf.zeros_initializer()
# # tf.random_uniform_initializer()
# # tf.random_normal_initializer()
# # tf.contrib.layers.xavier_initializer()

import tensorflow as tf
import matplotlib.pyplot as plt
import random

# Dataset
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
mnist = read_data_sets('MNIST_data/', one_hot=True)

# Input Layer
X = tf.placeholder(tf.float32, [None, 28*28])
Y = tf.placeholder(tf.float32, [None, 10])

# Hidden Layer
class hidden_input:
    W = None
    b = None
    x = None
    y1 = None
    y2 = None

    def __init__(self, x, input_node, ouput_node):
        self.W = tf.Variable(tf.random_normal([input_node, ouput_node]))
        self.b = tf.Variable(tf.random_normal([ouput_node]))
        logics = tf.matmul(x, self.W) + self.b
        self.y1 = tf.nn.sigmoid(logics)
        self.y2 = tf.nn.relu(self.y1)

class hidden_Layer:

    W = None
    b = None
    L1 = None
    L2 = None

    def __init__(self, x, input_node, ouput_node, keep_prob, name = 'w', initialize = tf.contrib.layers.xavier_initializer()):
        self.W = tf.get_variable(name, shape = [input_node, ouput_node], initializer = initialize)
        self.b = tf.Variable(tf.random_normal([ouput_node]))
        
        logics = tf.matmul(x, self.W) + self.b
        if keep_prob <= 0:
            self.L1 = tf.nn.relu(logics)
            self.L2 = tf.nn.softmax(self.L1)
        else:
            self.L1 = tf.nn.relu(logics)
            self.L2 = tf.nn.dropout(self.L1, keep_prob)


h1 = hidden_input(X, 28*28, 25)
h2 = hidden_Layer(h1.y2, 25, 50, 0.5,'w2')
h3 = hidden_Layer(h2.L2, 50, 50, 0.5, 'w3')
h4 = hidden_Layer(h3.L2, 50, 10, 0,'w4')

w = h4.W
hypothesis = h4.L2

# loss function
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Test model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters
num_epochs = 30
batch_size = 100
num_iterations = int(mnist.train.num_examples / batch_size)

with tf.Session() as sess:
    # Initializes global variables in the graph
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(num_epochs):
        avg_cost = 0

        for i in range(num_iterations):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, cost_val = sess.run([train, cost], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += cost_val / num_iterations

        print("Epoch: {:04d}, Cost: {:.9f}".format(epoch + 1, avg_cost))

    print("Learning finished")

    # Test the model using test sets
    print("Accuracy: ", accuracy.eval(
        session=sess, feed_dict={X: mnist.test.images, Y:mnist.test.labels}
        ),
    )

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r : r + 1], 1)))
    print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r : r + 1]}),
    )

    plt.imshow(
        mnist.test.images[r : r + 1].reshape(28, 28),
        cmap="Greys",
        interpolation="nearest"
    )
    plt.show()