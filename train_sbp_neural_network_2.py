import network
from train_test_helper_funcs import get_train_test_split_2, get_training_data, quantize_float
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
    train_pids_list, test_pids_list = get_train_test_split_2()

    fold = 1

    for train_pids in train_pids_list:
        print('{}-fold cross validation fold {}'.format(len(train_pids_list), fold))

        nn_model_save_filename = train_test_helper_funcs.neural_network_model_save_filename.strip().split('.')[0] + '_fold_' + str(fold) + '_.json'

        print('DEBUG - Getting training data...')

        ini_time = time.time()

        training_data = get_training_data(train_pids)
        print('DEBUG - Got training data')

        print('INFO - Execution time for getting training data: ' + str(quantize_float(time.time() - ini_time)) + ' s')

        nn = network.Network([num_inp_layer_neurons, num_hidden_layer_neurons, num_out_layer_neurons])
        try:
            nn.SGD(training_data, epochs, mini_batch_size, eta, lmbda=lmbda, monitor_training_cost=True, monitor_training_accuracy=True)
        except KeyboardInterrupt:
            nn.save(nn_model_save_filename)
            exit(0)

        nn.save(nn_model_save_filename)

        fold += 1