import numpy as np
import network
from train_test_helper_funcs import get_train_test_split, get_test_data, get_prediction, compute_save_prediction_results, quantize_float
import time

if __name__ == '__main__':
    train_pids, test_pids = get_train_test_split()

    nn = network.load('network_params.json')

    print('DEBUG - Computing and saving prediction results...')

    ini_time = time.time()

    mape = compute_save_prediction_results(nn)
    print('DEBUG - Computed and saved prediction results to prediction_results.txt')

    print('INFO - Execution time for computing and saving prediction results: ' + str(quantize_float(time.time() - ini_time)) + ' s')

    print('INFO - Mean Absolute percentage error (MAPE) = ' + str(mape))

    # print('DEBUG - Getting test data...')
    test_input, test_result = get_test_data(test_pids)
    # print('DEBUG - Got test data...')

    print('Actual SBP = ' + str(test_result))
    prediction = get_prediction(nn, test_input)
    print('Predicted SBP = ' + str(prediction)
    print('INFO - Absolute Percentage error = ' + str(quantize_float(abs(prediction - test_result) / test_result * 100)))