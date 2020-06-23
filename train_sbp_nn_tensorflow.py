import tensorflow as tf
import numpy as np
from train_test_helper_funcs_tensorflow import get_train_test_split, get_training_data, vectorized_result_list, quantize_float
import time
import sys

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

def init_weights(shape):
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    h = tf.nn.sigmoid(tf.matmul(X, w_1))
    # h = tf.nn.relu(tf.matmul(X, w_1))
    yhat = tf.matmul(h, w_2)
    return yhat

def main():
    tf.get_logger().setLevel('ERROR')

    np.set_printoptions(threshold=sys.maxsize)

    train_pids, test_pids = get_train_test_split()

    print('DEBUG - Getting training data...')
    
    ini_time = time.time()

    train_X, train_y = get_training_data(train_pids)
    # train_X = np.array([[1.0, 65, 30], [1.0, 75, 45]], dtype=np.float64)
    # train_y = np.array([vectorized_result_list(115), vectorized_result_list(125)], dtype=np.float64)
    # train_y = np.array([[115 / 250], [125 / 250]], dtype=np.float64)
    print('DEBUG - Got training data')

    print('INFO - Execution time for getting training data: ' + str(quantize_float(time.time() - ini_time)) + ' s')

    # print('DEBUG - train_y = ' + str(train_y))

    x_size = train_X.shape[1]
    h_size = 10
    y_size = train_y.shape[1]
    # print('DEBUG - x_size: ' + str(x_size))
    # print('DEBUG - y_size: ' + str(y_size))

    X = tf.placeholder("float", shape=[None, x_size], name='X')
    y = tf.placeholder("float", shape=[None, y_size], name='y')

    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    yhat = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1, name='predict')
    # predict = tf.reduce_max(yhat, axis=1, name='predict')

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    # updates = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    updates = tf.train.AdamOptimizer(0.001).minimize(cost)

    saver = tf.train.Saver()

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # start = 0
    # end = int(len(train_X) / 10)

    for epoch in range(10):
        print('DEBUG - Beginning epoch ' + str(epoch + 1) + '...')

        ini_time = time.time()

        try:
            # for i in range(start, end):
            for i in range(len(train_X)):
                sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})
        except KeyboardInterrupt:
            print('DEBUG - Epoch ' + str(epoch + 1) + ' interrupted')
            saver.save(sess, 'nn_tensorflow')
            sess.close()
            exit(0)

        print('DEBUG - Epoch ' + str(epoch + 1) + ' ended')

        print('INFO - Execution time for epoch ' + str(epoch + 1) + ': ' + str(quantize_float(time.time() - ini_time)) + ' s')

        train_accuracy = np.mean(np.argmax(train_y, axis=1) == sess.run(predict, feed_dict={X: train_X, y: train_y}))
        # train_accuracy = np.mean(np.max(train_y, axis=1) == sess.run(predict, feed_dict={X: train_X, y: train_y}))

        print('DEBUG - actual: ' + str(np.argmax(train_y, axis=1)))
        # print('DEBUG - actual: ' + str(np.max(train_y, axis=1)))
        print('DEBUG - prediction: ' + str(sess.run(predict, feed_dict={X: train_X, y: train_y})))

        print('Epoch = %d, train accuracy = %.2f%%' % (epoch + 1, 100 * train_accuracy))

        # start += int((len(train_X) / 10))
        # end += int((len(train_X) / 10))

    saver.save(sess, 'nn_tensorflow')

    sess.close()

if __name__ == '__main__':
    main()