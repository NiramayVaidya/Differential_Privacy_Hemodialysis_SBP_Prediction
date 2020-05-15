import tensorflow as tf
import numpy as np
from train_test_helper_funcs_tensorflow import get_train_test_split, get_test_data, compute_save_prediction_results, quantize_float
import time

def main():
    tf.get_logger().setLevel('ERROR')

    train_pids, test_pids = get_train_test_split()

    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.import_meta_graph('nn_tensorflow.meta')
        saver.restore(sess, tf.compat.v1.train.latest_checkpoint('./'))

        graph = tf.compat.v1.get_default_graph()

        X = graph.get_tensor_by_name('X:0')
        y = graph.get_tensor_by_name('y:0')
        predict = graph.get_tensor_by_name('predict:0')
    
        print('DEBUG - Computing and saving prediction results...')

        ini_time = time.time()

        compute_save_prediction_results(test_pids, sess, X, y, predict)
        print('DEBUG - Computed and saved prediction results to prediction_results.txt')

        print('INFO - Execution time for computing and saving prediction results: ' + str(quantize_float(time.time() - ini_time)) + ' s')
    
        # print('DEBUG - Getting test data...')
        test_X, test_y = get_test_data(test_pids)
        # print('DEBUG - Got test data')

        # actual_sbp = np.argmax(test_y, axis=1)[0] + 1
        actual_sbp = test_y[0][0]
        print('Actual SBP = ' + str(actual_sbp))
        # predicted_sbp = sess.run(predict, feed_dict={X: test_X, y: test_y})[0] + 1
        predicted_sbp = sess.run(predict, feed_dict={X: test_X, y: test_y})[0] * 250
        print('Predicted SBP = ' + str(predicted_sbp))
        print('INFO - Percentage error = ' + str(abs(predicted_sbp - actual_sbp) / actual_sbp * 100))

if __name__ == '__main__':
    main()