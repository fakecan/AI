import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)    # constant: 고정
node2 = tf.constant(4.0)    # default tf.
node3 = tf.add(node1, node2)

# print("node1 : ", node1, "node2 : ", node2)
# print("node3 : ", node3)

sess = tf.Session()
# print("sess.run(node1, node2) : ", sess.run([node1, node2]))  # 노드 성분
# print("sess.run(node3) : ", sess.run(node3))

a = tf.placeholder(tf.float32)  # placeholder + feed_dict
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))   # feed_dict: 직접 값을 넣는다
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))

add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, feed_dict={a: 3, b: 4.5}))