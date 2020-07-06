import tensorflow as tf
import numpy as np
from train_test_helper_funcs_tensorflow import get_train_test_split_2, get_test_data, compute_save_prediction_results_2, quantize_float
import train_test_helper_funcs_tensorflow
import time

def main():
    tf.get_logger().setLevel('ERROR')

    train_pids_list, test_pids_list = get_train_test_split_2()

    fold = 1
    avg_mape = 0

    for test_pids in test_pids_list:
        print('{}-fold cross validation fold {}'.format(len(test_pids_list), fold))

        nn_tensorflow_model_save_filename = train_test_helper_funcs.nn_tensorflow_model_save_filename.strip() + '_fold_' + str(fold) + '_.meta'

        with tf.compat.v1.Session() as sess:
            saver = tf.compat.v1.train.import_meta_graph(nn_tensorflow_model_save_filename)
            saver.restore(sess, tf.compat.v1.train.latest_checkpoint('saved_models_nn_tensorflow/'))

            graph = tf.compat.v1.get_default_graph()

            X = graph.get_tensor_by_name('X:0')
            y = graph.get_tensor_by_name('y:0')
            predict = graph.get_tensor_by_name('predict:0')
    
            print('DEBUG - Computing and saving prediction results...')

            ini_time = time.time()

            mape = compute_save_prediction_results_2(test_pids, sess, X, y, predict, fold)
            avg_mape += mape
            print('DEBUG - Computed and saved prediction results to ' + train_test_helper_funcs_tensorflow.prediction_results_filename)

            print('INFO - Execution time for computing and saving prediction results: ' + str(quantize_float(time.time() - ini_time)) + ' s')

            print('INFO - Mean Absolute percentage error (MAPE) = ' + str(mape))
    
            if fold == len(test_pids_list):
                # print('DEBUG - Getting test data...')
                # test_X, test_y = get_test_data(test_pids)
                # print('DEBUG - Got test data')

                # actual_sbp = np.argmax(test_y, axis=1)[0] + 1
                # actual_sbp = test_y[0][0] * 250
                # print('Actual SBP = ' + str(actual_sbp))
                # predicted_sbp = sess.run(predict, feed_dict={X: test_X, y: test_y})[0] + 1
                # predicted_sbp = quantize_float(sess.run(predict, feed_dict={X: test_X, y: test_y})[0] * 250)
                # print('Predicted SBP = ' + str(predicted_sbp))
                # print('INFO - Absolute Percentage error = ' + str(quantize_float(abs((predicted_sbp - # actual_sbp) / actual_sbp * 100))))

        fold += 1

    avg_mape /= len(test_pids_list)
    print('INFO - Average MAPE = ' + str(quantize_float(avg_mape)))

if __name__ == '__main__':
    main()