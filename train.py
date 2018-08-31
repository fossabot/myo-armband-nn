from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf

from time import time
from include.data import get_data_set
from include.model import model
from tensorflow.python.saved_model.builder_impl import SavedModelBuilder

tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
FLAGS = tf.app.flags.FLAGS

train_x, train_y = get_data_set()

_BATCH_SIZE = 300
_CLASS_SIZE = 6
_SAVE_PATH = "./data/tensorflow_sessions/myo_armband/"

x, y, output, global_step, y_pred_cls = model(_CLASS_SIZE)


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
train_writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph)


try:
    print("Trying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
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
            saver.save(sess, save_path=_SAVE_PATH, global_step=global_step)
            print("Saved checkpoint.")


train(7500)

tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
tensor_info_y = tf.saved_model.utils.build_tensor_info(y)

table = tf.contrib.lookup.index_to_string_table_from_tensor(tf.constant([str(i) for i in range(_CLASS_SIZE)]))
values, indices = tf.nn.top_k(y, _CLASS_SIZE)
prediction_classes = table.lookup(tf.to_int64(indices))


classification_inputs = tf.saved_model.utils.build_tensor_info(serialized_tf_example)
classification_outputs_classes = tf.saved_model.utils.build_tensor_info(prediction_classes)
classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)

classification_signature = (
  tf.saved_model.signature_def_utils.build_signature_def(
      inputs={
          tf.saved_model.signature_constants.CLASSIFY_INPUTS:
              classification_inputs
      },
      outputs={
          tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
              classification_outputs_classes,
          tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
              classification_outputs_scores
      },
      method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

prediction_signature = (
  tf.saved_model.signature_def_utils.build_signature_def(
      inputs={'images': tensor_info_x},
      outputs={'scores': tensor_info_y},
      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

export_path_base = "/script-1/data"
export_path = os.path.join(
      tf.compat.as_bytes(export_path_base),
      tf.compat.as_bytes(str(FLAGS.model_version)))
print("Exporting trained model to", export_path)
builder = tf.saved_model.builder.SavedModelBuilder(export_path)


builder.add_meta_graph_and_variables(
  sess, [tf.saved_model.tag_constants.SERVING],
  signature_def_map={
      'predict':
          prediction_signature,
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          classification_signature,
  },
  main_op=tf.tables_initializer(),
  strip_default_attrs=True)

sess.close()
