# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 기본 모델 ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
import tensorflow as tf
# tf.set_random_seed(777)

x_data = []
y_data = []

# Placeholders
X = tf.placeholder(tf.float32, shape=[None, input])
Y = tf.placeholder(tf.float32, shape=[None, output])

W = tf.Variable(tf.random_normal([input, output]), name='weight')
b = tf.Variable(tf.random_normal([output]), name='bias')


# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 회귀(Regressor) ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Cost/Loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Sigmoid
# hypothesis = tf.sigmoid(tf.matmul(X, W) + b)    # 0과 1 사이의 값
# cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
# train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
# predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))    # 일반적인 선형 회귀에서는 안된다

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 분류(Classification) ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1)) # categorical_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Correct prediction Test model
prediction = tf.argmax(hypothesis, 1)
is_correct= tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))



# Launch graph
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    for step in range(1001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis:\n", h, "\nCorrect (Y):\n", c, "\nAccuracy: ", a)

# for step in range(1001):
#     cost_val, hy_val, _ = sess.run(
#         [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
#     print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)



# # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ sess.run 종류 ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# # 1. sess.run
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# aaa = sess.run(W)
# print(aaa)
# sess.close()

# # 2. tf.InteractiveSession(), eval()
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# aaa = W.eval()  # sess.run과 같은 표현이다
# print(aaa)
# sess.close()

# # 3. eval(session=sess)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# aaa = W.eval(session=sess)  # sess.run과 같은 표현이다
# print(aaa)
# sess.close()
# # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
