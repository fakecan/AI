# Multi-variable linear regression 1
import tensorflow as tf
tf.set_random_seed(777) # for reproducibility

x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]

y_data = [152., 185., 180., 196., 142.]

# placeholders for a tensor that will be always fed.
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize. Need a very small learning rate for this data set
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Inititalizes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(2000001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                    feed_dict={x1: x1_data, x2: x2_data,
                                               x3: x3_data, Y: y_data})
    if step % 10 == 0:
        print("Step: ", step, ", Cost: ", cost_val, ", Prediction: ", hy_val)