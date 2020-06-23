import network
from train_test_helper_funcs import get_train_test_split, get_training_data, quantize_float
import time
import sys

epochs = 30
mini_batch_size = 10
eta = 0.1
lmbda = 0.1
num_inp_layer_neurons = 2
num_hidden_layer_neurons = 50
num_out_layer_neurons = 250

if __name__ == '__main__':
    train_pids, test_pids = get_train_test_split()

    print('DEBUG - Getting training data...')

    ini_time = time.time()

    training_data = get_training_data(train_pids)
    print('DEBUG - Got training data')

    print('INFO - Execution time for getting training data: ' + str(quantize_float(time.time() - ini_time)) + ' s')

    nn = network.Network([num_inp_layer_neurons, num_hidden_layer_neurons, num_out_layer_neurons])
    try:
        nn.SGD(training_data, epochs, mini_batch_size, eta, lmbda=lmbda, monitor_training_cost=True, monitor_training_accuracy=True)
    except KeyboardInterrupt:
        nn.save('network_params.json')
        exit(0)
    nn.save('network_params.json')