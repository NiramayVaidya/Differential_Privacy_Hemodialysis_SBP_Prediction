import tensorflow as tf
import numpy as np
from train_test_helper_funcs_tensorflow import get_train_test_split, get_training_data, vectorized_result_list

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

def init_weights(shape):
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    h = tf.nn.sigmoid(tf.matmul(X, w_1))
    yhat = tf.matmul(h, w_2)
    return yhat

def main():
    tf.get_logger().setLevel('ERROR')

    train_pids, test_pids = get_train_test_split()

    print('DEBUG - Getting training data...')
    train_X, train_y = get_training_data(train_pids)
    # train_X = np.array([[1.0, 65, 30], [1.0, 75, 45]], dtype=np.float64)
    # train_y = np.array([vectorized_result_list(115), vectorized_result_list(125)], dtype=np.float64)
    print('DEBUG - Got training data')

    x_size = train_X.shape[1]
    h_size = 10
    y_size = train_y.shape[1]

    X = tf.placeholder("float", shape=[None, x_size], name='X')
    y = tf.placeholder("float", shape=[None, y_size], name='y')

    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    yhat = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1, name='predict')

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    saver = tf.train.Saver()

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(10):
        print('DEBUG - Beginning epoch ' + str(epoch + 1) + '...')
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})
        print('DEBUG - Epoch ' + str(epoch + 1) + ' ended')

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_X, y: train_y}))

        print('Epoch = %d, train accuracy = %.2f%%' % (epoch + 1, 100.0 * train_accuracy))

    saver.save(sess, 'nn_tensorflow')

    sess.close()

if __name__ == '__main__':
    main()