import time
from train_test_helper_funcs_regression import get_train_test_split_2, get_test_data, predict, compute_save_prediction_results_2, quantize_float
from sklearn.externals import joblib
# import pickle

def main():
    train_pids_list, test_pids_list = get_train_test_split_2()

    regressors = joblib.load('regression_2.sav')
    # regressors = pickle.load(open('regression_2.sav', 'rb'))

    fold = 1
    index = 0
    avg_mape = 0

    for regressor in regressors:
        print('{}-fold cross validation fold {}'.format(len(regressors), fold))

        print('DEBUG - Computing and saving prediction results...')

        ini_time = time.time()

        mape = compute_save_prediction_results_2(regressor, test_pids_list[index], fold)
        avg_mape += mape
        print('DEBUG - Computed and saved prediction results to prediction_results_regression_2.txt')

        print('INFO - Execution time for computing and saving prediction results: ' + str(quantize_float(time.time() - ini_time)) + ' s')

        print('INFO - Mean Absolute percentage error (MAPE) = ' + str(mape))

        fold += 1
        index += 1

    avg_mape /= len(regressors)
    print('INFO - Average MAPE = ' + str(quantize_float(avg_mape)))

    # print('DEBUG - Getting test data...')
    # x_test, y_expect = get_test_data(test_pids_list[index])
    # print('DEBUG - Got test data...')

    # print('Actual SBP = ' + str(y_expect))
    # y_pred = predict(regressor, x_test)
    # print('Predicted SBP = ' + str(y_pred))
    # print('INFO - Absolute Percentage error = ' + str(quantize_float(abs(y_pred - y_expect) / y_expect * 100)))

if __name__ == '__main__':
    main()