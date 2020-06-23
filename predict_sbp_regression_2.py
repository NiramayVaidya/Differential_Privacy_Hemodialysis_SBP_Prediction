import time
from train_test_helper_funcs_regression import get_train_test_split, get_test_data, predict, compute_save_prediction_results, quantize_float
from sklearn.externals import joblib
# import pickle

def main():
    train_pids, test_pids = get_train_test_split()

    regressor = joblib.load('regression.sav')
    # regressor = pickle.load(open('regression.sav', 'rb'))

    print('DEBUG - Computing and saving prediction results...')

    ini_time = time.time()

    mape = compute_save_prediction_results(regressor, test_pids)
    print('DEBUG - Computed and saved prediction results to prediction_results_regression.txt')

    print('INFO - Execution time for computing and saving prediction results: ' + str(quantize_float(time.time() - ini_time)) + ' s')

    print('INFO - Mean Absolute percentage error (MAPE) = ' + str(mape))

    # print('DEBUG - Getting test data...')
    x_test, y_expect = get_test_data(test_pids)
    # print('DEBUG - Got test data...')

    print('Actual SBP = ' + str(y_expect))
    y_pred = predict(regressor, x_test)
    print('Predicted SBP = ' + str(y_pred))
    print('INFO - Absolute Percentage error = ' + str(quantize_float(abs(y_pred - y_expect) / y_expect * 100)))

if __name__ == '__main__':
    main()