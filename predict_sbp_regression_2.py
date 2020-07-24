import time
from train_test_helper_funcs_regression import get_train_test_split, get_test_data, predict, predict_mi_attack_dbp, compute_save_prediction_results, quantize_float
import train_test_helper_funcs_regression
from sklearn.externals import joblib
# import pickle

def main():
    train_pids, test_pids = get_train_test_split()

    regressor = joblib.load(train_test_helper_funcs_regression.regression_model_save_filename)
    # regressor = pickle.load(open(train_test_helper_funcs_regression.regression_model_save_filename, 'rb'))

    print('DEBUG - Computing and saving prediction results...')

    ini_time = time.time()

    mape, mi_attack_dbp_mape = compute_save_prediction_results(regressor, test_pids)
    # print('DEBUG - Computed and saved prediction results to ' + train_test_helper_funcs_regression.prediction_results_filename)
    print('DEBUG - Computed and saved MI attack DBP prediction results to ' + train_test_helper_funcs_regression.mi_attack_dbp_prediction_results_filename)

    print('INFO - Execution time for computing and saving prediction results: ' + str(quantize_float(time.time() - ini_time)) + ' s')

    # print('INFO - Mean Absolute percentage error (MAPE) = ' + str(mape))
    print('INFO - MI Attack DBP Mean Absolute percentage error (MAPE) = ' + str(mi_attack_dbp_mape))

    # print('DEBUG - Getting test data...')
    # x_test, y_expect = get_test_data(test_pids)
    # print('DEBUG - Got test data...')

    # print('Actual SBP = ' + str(y_expect))
    # y_pred = predict(regressor, x_test)
    # print('Predicted SBP = ' + str(y_pred))
    # print('INFO - Absolute Percentage error = ' + str(quantize_float(abs(y_pred - y_expect) / y_expect * 100)))
    # print('INFO - Absolute Percentage error = ' + str(quantize_float(abs((y_pred - y_expect) / y_expect * 100))))

    # print('Actual DBP = ' + str(x_test[0]))
    # mi_attack_dbp_pred = predict_mi_attack_dbp(regressor, y_pred, x_test[1])
    # print('Predicted DBP = ' + str(mi_attack_dbp_pred))
    # print('INFO - MI Attack DBP Absolute Percentage error = ' + str(quantize_float(abs((mi_attack_dbp_pred - x_test[0]) / x_test[0] * 100))))

if __name__ == '__main__':
    main()