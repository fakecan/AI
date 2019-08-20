import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score #레거시한 머신 러닝 중 하나

tf.set_random_seed(777)

xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973], 
               [823.02002, 828.070007, 1828100, 821.655029, 828.070007], 
               [819.929993, 824.400024, 1438100, 818.97998, 824.159973], 
               [816, 820.958984, 1008100, 815.48999, 819.23999], 
               [819.359985, 823, 1188100, 818.469971, 818.97998], 
               [819, 823, 1198100, 816, 820.450012], 
               [811.700012, 815.25, 1098100, 809.780029, 813.669983], 
               [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
print(x_data.shape, y_data.shape)   # (8, 4) (8, 1)
# x_data = x_data('float32')
# y_data = y_data('float32')

X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 1])

W = tf.Variable(tf.random_normal([4, 1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
train = tf.train.AdadeltaOptimizer(learning_rate=0.001).minimize(cost)

# Launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

for step in range(5001):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

#RMSE 구하기
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_data, hy_val))

# R2 구하기
r2_y_predict = r2_score(y_data, hy_val)
print("R2 : ", r2_y_predict)