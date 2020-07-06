import numpy as np
import network
from train_test_helper_funcs import get_train_test_split_2, get_test_data, get_prediction, compute_save_prediction_results_2, quantize_float
import train_test_helper_funcs
import time

if __name__ == '__main__':
    train_pids_list, test_pids_list = get_train_test_split_2()

    fold = 1
    avg_mape = 0

    for test_pids in test_pids_list:
        print('{}-fold cross validation fold {}'.format(len(test_pids_list), fold))

        nn_model_save_filename = train_test_helper_funcs.neural_network_model_save_filename.strip().split('.')[0] + '_fold_' + str(fold) + '_.json'

        nn = network.load(nn_model_save_filename)

        print('DEBUG - Computing and saving prediction results...')

        ini_time = time.time()

        mape = compute_save_prediction_results_2(nn, test_pids, fold)
        avg_mape += mape
        print('DEBUG - Computed and saved prediction results to ' + train_test_helper_funcs.prediction_results_filename)

        print('INFO - Execution time for computing and saving prediction results: ' + str(quantize_float(time.time() - ini_time)) + ' s')

        print('INFO - Mean Absolute percentage error (MAPE) = ' + str(mape))

        if fold == len(test_pids_list):
            # print('DEBUG - Getting test data...')
            # test_input, test_result = get_test_data(test_pids)
            # print('DEBUG - Got test data...')

            # print('Actual SBP = ' + str(test_result))
            # prediction = get_prediction(nn, test_input)
            # print('Predicted SBP = ' + str(prediction))
            # print('INFO - Absolute Percentage error = ' + str(quantize_float(abs((prediction - # test_result) / test_result * 100))))
        
        fold += 1

    avg_mape /= len(test_pids_list)
    print('INFO - Average MAPE = ' + str(quantize_float(avg_mape)))