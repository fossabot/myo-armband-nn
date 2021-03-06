from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import shutil
import numpy as np
import tensorflow as tf

from time import time
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model.builder_impl import SavedModelBuilder
from pathlib import Path
from data import get_data_set
from model import model
from common import *

_BATCH_SIZE = 300
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
FLAGS = tf.app.flags.FLAGS

train_x, train_y = get_data_set(get_train_data_location())

save_path = str(Path(get_tf_session_dir())) + os.sep
export_path_base = str(Path(get_tf_export_dir())) + os.sep

x, y, output, global_step, y_pred_cls = model()


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
tf.summary.scalar("Loss", loss)
optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)


correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, dimension=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("Accuracy/train", accuracy)


init = tf.global_variables_initializer()
merged = tf.summary.merge_all()
saver = tf.train.Saver()
sess = tf.Session()
serialized_tf_example = tf.placeholder(tf.string, name='myo-nn')
train_writer = tf.summary.FileWriter(save_path, sess.graph)


try:
    print("Trying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_path)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except:
    print("Failed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())


def train(num_iterations = 1000):
    for i in range(num_iterations):
        randidx = np.random.randint(len(train_x), size=_BATCH_SIZE)
        batch_xs = train_x[randidx]
        batch_ys = train_y[randidx]
        start_time = time()
        i_global, _ = sess.run([global_step, optimizer], feed_dict={x: batch_xs, y: batch_ys})
        duration = time() - start_time

        if (i_global % 10 == 0) or (i == num_iterations - 1):
            _loss, batch_acc = sess.run([loss, accuracy], feed_dict={x: batch_xs, y: batch_ys})
            msg = "Global Step: {0:>6}, accuracy: {1:>6.1%}, loss = {2:.2f} ({3:.1f} examples/sec, {4:.2f} sec/batch)"
            print(msg.format(i_global, batch_acc, _loss, _BATCH_SIZE / duration, duration))

        if (i_global % 100 == 0) or (i == num_iterations - 1):
            data_merged, global_1 = sess.run([merged, global_step], feed_dict={x: batch_xs, y: batch_ys})
            train_writer.add_summary(data_merged, global_1)
            saver.save(sess, save_path=save_path, global_step=global_step)
            print("Saved checkpoint.")


train(30000)

export_path = os.path.join(
      tf.compat.as_bytes(export_path_base),
      tf.compat.as_bytes(str(FLAGS.model_version)))
if os.path.exists(export_path) and os.path.isdir(export_path):
    shutil.rmtree(export_path)
print("Exporting trained model to", export_path)

with sess.graph.as_default():
    prediction_signature = signature_def_utils.build_signature_def(
        inputs={
            "input": utils.build_tensor_info(x)
        },
        outputs={
            "output": utils.build_tensor_info(y_pred_cls),
            "distr": utils.build_tensor_info(output)
        },
        method_name=signature_constants.PREDICT_METHOD_NAME
    )
    builder = saved_model_builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            "serving_default": prediction_signature,
            "predict": prediction_signature
        })
    builder.save()

sess.close()
