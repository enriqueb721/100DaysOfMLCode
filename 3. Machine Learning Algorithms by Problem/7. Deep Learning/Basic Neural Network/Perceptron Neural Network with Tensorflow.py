import numpy as np
import tensorflow as tf

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

dX = tf.placeholder(tf.float32, shape=[None, 2], name="X")
dY = tf.placeholder(tf.float32, shape=[None, 2], name="Y")


def linear(input_, out_dim, scope):
    with tf.variable_scope(scope):
        W = tf.get_variable("W", shape=[input_.get_shape()[1], out_dim], dtype=input_.dtype)
        b = tf.get_variable("b", shape=[out_dim], dtype=input_.dtype)
    return tf.add(tf.matmul(input_, W), b)


def softmax(input_, out_dim, scope):
    with tf.variable_scope(scope):
        W = tf.get_variable("W", shape=[input_.get_shape()[1], out_dim], dtype=input_.dtype)
        b = tf.get_variable("b", shape=[out_dim], dtype=input_.dtype)
    return tf.nn.softmax(tf.add(tf.matmul(input_, W), b))


def model(input_, activation=tf.nn.relu):
    with tf.variable_scope("model"):
        l1 = activation(linear(input_, 20, "l1"))
        l2 = activation(linear(l1, 20, "l2"))
        m = softmax(l2, 2, "final")
    return m

pred = model(dX)

# regression
cost = tf.reduce_mean(tf.pow(dY - pred, 2))
# classification
cost = - tf.reduce_sum(dY * tf.log(tf.clip_by_value(pred, 1e-10, 10)))

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
epochs = 5000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    current_epoch = 0
    while current_epoch < epochs:
        current_epoch += 1
        _, loss = sess.run([optimizer, cost], feed_dict={dX: x, dY: y})
        print("current_epoch", current_epoch, "loss", loss)
    
    a = sess.run(pred, feed_dict={dX: x})
    print(a)