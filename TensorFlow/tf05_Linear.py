import tensorflow as tf
tf.set_random_seed(777) # 같은 랜덤 값

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

#   constant와 placeholder 구분
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = x_train * W + b

# ■■■■■■■■■■ --- model.compile --- ■■■■■■■■■■
#   coss/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))  # (loss='mse', optimizer='adam')

# optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
#   경사하강
# ■■■■■■■■■■ -------------------- ■■■■■■■■■■

# Launch the graph in a session.
with tf.Session() as sess:  # with를 씀으로서 close 안해도 된다
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer()) # ★ 초기화(변수)

    # Fit the line
    for step in range(2001):    # epochs
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b])   # sess.run은 fit이다
        #   _ 자리에 train 들어가고 순서대로

        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)
