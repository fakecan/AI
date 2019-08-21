

# import tensorflow as tf
# import matplotlib.pyplot as plt
# import random
# from tensorflow.examples.tutorials.mnist import input_data

# tf.set_random_seed(777)

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# # print(mnist.train.images.shape)     # (55000, 784)
# # print(mnist.test.labels.shape)      # (10000, 10)





# nb_classes = 10
# keep_prob = 0.2

# X = tf.placeholder(tf.float32, [None, 28*28])
# Y = tf.placeholder(tf.float32, [None, 10])

# W1 = tf.Variable(tf.random_normal([28*28, 30]))
# b1 = tf.Variable(tf.random_normal([30]))
# layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
# layer1 = tf.nn.dropout(layer1, keep_prob)

# W2 = tf.Variable(tf.random_normal([30, nb_classes]))
# b2 = tf.Variable(tf.random_normal([nb_classes]))
# layer2 = tf.nn.relu(tf.matmul(X, W1) + b1)
# layer2 = tf.nn.dropout(layer2, keep_prob)



import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# print(mnist.train.images.shape)     # (55000, 784)
# print(mnist.test.labels.shape)      # (10000, 10)

nb_classes = 10

X = tf.placeholder(tf.float32, [None, 28*28])
Y = tf.placeholder(tf.float32, [None, 10])

# W = tf.Variable(tf.random_normal([28*28, nb_classes]))
# b = tf.Variable(tf.random_normal([nb_classes]))

class hiddenLayer:
    X = None
    Y1 = None
    Y2 = None
    W = None
    b = None

    def __init__(self, X, input, output):
        self.W = tf.Variable(tf.random_normal([input, output]))
        self.b = tf.Variable(tf.random_normal([output]))
        equation = tf.matmul(X, self.W) + self.b
        self.Y1 = tf.nn.softmax(equation)
        self.Y2 = tf.nn.relu(equation)

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

class hiddenLayer2:
    X = None
    Y1 = None
    Y2 = None
    W = None
    b = None

    def __init__(self, X, input, output, name="W"):
        self.W = tf.get_variable(name, shape = [input, output], initializer=tf.contrib.layers.xavier_initializer())
        self.b = tf.Variable(tf.random_normal([output]))
        equation = tf.matmul(X, self.W) + self.b
        self.Y1 = tf.nn.softmax(equation)
        self.Y2 = tf.nn.relu(equation)

h1 = hiddenLayer(X, 28*28, 16)
h2 = hiddenLayer2(h1.Y2, 16, 8)
h3 = hiddenLayer2(h2.Y2, 8, 64)
h4 = hiddenLayer2(h3.Y2, 64, 12)
h5 = hiddenLayer2(h4.Y2, 12, 2)
h6 = hiddenLayer2(h5.Y2, 2, 8)
h7 = hiddenLayer2(h6.Y2, 8, 16)
h8 = hiddenLayer2(h7.Y2, 16, 8)
h9 = hiddenLayer2(h8.Y2, 8, 4)
h10 = hiddenLayer(h9.Y1, 4, nb_classes)

W = h10.W
hypothesis = h10.Y1

# hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Test model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters
num_epochs = 10
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