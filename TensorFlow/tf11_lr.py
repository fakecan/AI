# Lab 7 Learning rate and Evaluation
import tensorflow as tf
tf.set_random_seed(777)

x_data = [[1, 2, 1], [1, 3, 2],
          [1, 3, 4], [1, 5, 5],
          [1, 7, 5], [1, 2, 5],
          [1, 6, 6], [1, 7, 7]]

y_data = [[0, 0, 1], [0, 0, 1],
          [0, 0, 1], [0, 1, 0],
          [0, 1, 0], [0, 1, 0],
          [1, 0, 0], [1, 0, 0]]

# Evaluatiomn our model using this test datasewt
x_test = [[2, 1, 1], [3, 1, 2], [3, 3, 4]]
y_test = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])

W = tf.Variable(tf.random_normal([3, 3]))
b = tf.Variable(tf.random_normal([3]))

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1)) # categorical_crossentropy
# Try to change learning_rate to small numbers
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.06).minimize(cost)

# Correct prediction Test model
prediction = tf.argmax(hypothesis, 1)
is_correct= tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(201):
        cost_val, W_val, _ = sess.run([cost, W, optimizer], feed_dict={X: x_data, Y: y_data})
        print(step, cost_val, W_val)

    # predict
    print("Prediction: ", sess.run(prediction, feed_dict={X: x_test}))
    # Calculate the accuracy
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))


# learning_rate=    0.1     |   0.08    |   0.01    |   0.001
# Prediction:       [2 2 2] |   [2 2 2] |   [0 0 1] |   [0 0 0]
# Accuracy:         1.0     |   1.0     |   0.0     |   0.0