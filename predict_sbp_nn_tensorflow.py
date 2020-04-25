import tensorflow as tf
import numpy as np
from train_test_helper_funcs_tensorflow import get_train_test_split, get_test_data

def main():
    tf.get_logger().setLevel('ERROR')

    train_pids, test_pids = get_train_test_split()

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('nn_tensorflow.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        graph = tf.get_default_graph()

        X = graph.get_tensor_by_name('X:0')
        y = graph.get_tensor_by_name('y:0')
        predict = graph.get_tensor_by_name('predict:0')
    
        print('DEBUG - Getting test data...')
        test_X, test_y = get_test_data(test_pids)
        print('DEBUG - Got test data')

        print('Actual SBP = ' + str(np.argmax(test_y, axis=1)[0] + 1))
        print('Predicted SBP = ' + str(sess.run(predict, feed_dict={X: test_X, y: test_y})[0] + 1))

if __name__ == '__main__':
    main()