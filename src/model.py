import tensorflow as tf
from common import *

def model():
    num_classes = get_num_classes()

    with tf.name_scope('data'):
        x = tf.placeholder(tf.float32, shape=[None, 64], name='Input')
        y = tf.placeholder(tf.float32, shape=[None, num_classes], name='Output')

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([64, 528])),
        'h2': tf.Variable(tf.random_normal([528, 786])),
        'h3': tf.Variable(tf.random_normal([786, 1248])),
        'out': tf.Variable(tf.random_normal([1248, num_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([528])),
        'b2': tf.Variable(tf.random_normal([786])),
        'b3': tf.Variable(tf.random_normal([1248])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    layer_3 = tf.nn.dropout(layer_3, 0.5)

    output = tf.add(tf.matmul(layer_3, weights['out']), biases['out'], name="output")

    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
    y_pred_cls = tf.argmax(output, dimension=1)

    return x, y, output, global_step, y_pred_cls