# 랜덤값으로 변수 1개를 만들고, 변수의 내용을 출력하시오.

import tensorflow as tf
tf.set_random_seed(777)

W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# v1 = sess.run(W)
# v2 = sess.run(b)
# print(v1, v2)

print(W)

W = tf.Variable([0.3], tf.float32)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# aaa = sess.run(W)
# print(aaa)
# sess.close()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
aaa = W.eval()  # sess.run과 같은 표현이다
print(aaa)
sess.close()