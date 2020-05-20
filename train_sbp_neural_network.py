import network
from train_test_helper_funcs import get_train_test_split, get_training_data, quantize_float
import time

epochs = 10
mini_batch_size = 10
eta = 0.5

if __name__ == '__main__':
    train_pids, test_pids = get_train_test_split()

    print('DEBUG - Getting training data...')

    ini_time = time.time()

    training_data = get_training_data(train_pids)
    print('DEBUG - Got training data')

    print('INFO - Execution time for getting training data: ' + str(quantize_float(time.time() - ini_time)) + ' s')

    nn = network.Network([2, 10, 250])
    nn.SGD(training_data, epochs, mini_batch_size, eta, monitor_training_cost=True, monitor_training_accuracy=True)
    nn.save('network_params.json')