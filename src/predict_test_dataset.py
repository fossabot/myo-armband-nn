import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from pathlib import Path

from model import model
from common import *
from data import get_data_set

save_path = Path(get_tf_session_dir())
x, y, output, global_step, y_pred_cls = model()


test_x, test_y = get_data_set(get_test_data_location())
test_l = ["Relax", "Ok", "Fist", "Like", "Rock", "Spock"]


saver = tf.train.Saver()
sess = tf.Session()


try:
    print("Trying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_path)
    print(last_chk_path)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except:
    print("Failed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())


i = 0
predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
while i < len(test_x):
    j = min(i + 300, len(test_x))
    batch_xs = test_x[i:j, :]
    batch_ys = test_y[i:j, :]
    predicted_class[i:j] = sess.run(y_pred_cls, feed_dict={x: batch_xs, y: batch_ys})
    i = j


correct = (np.argmax(test_y, axis=1) == predicted_class)
acc = correct.mean()*100
correct_numbers = correct.sum()
print("Accuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_x)))


cm = confusion_matrix(y_true=np.argmax(test_y, axis=1), y_pred=predicted_class)
for i in range(get_num_classes() - 1):
    class_name = "({}) {}".format(i, test_l[i])
    print(cm[i, :], class_name)
class_numbers = [" ({0})".format(i) for i in range(6)]
print("".join(class_numbers))


sess.close()
