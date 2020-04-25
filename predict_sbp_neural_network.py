import numpy as np
import network
from train_test_helper_funcs import get_train_test_split, get_test_data

def get_prediction(nn):
    return np.argmax(nn.feedforward(test_input)) + 1

if __name__ == '__main__':
    train_pids, test_pids = get_train_test_split()

    nn = network.load('network_params.json')

    print('DEBUG - Getting test data...')
    test_input, test_result = get_test_data(test_pids)
    print('DEBUG - Got test data...')

    print('True SBP = ' + str(test_result))
    print('Predicted SBP = ' + str(get_prediction(nn)))