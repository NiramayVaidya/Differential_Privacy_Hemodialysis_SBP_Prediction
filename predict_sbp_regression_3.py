import time
from train_test_helper_funcs_regression import get_train_test_split_2, get_test_data, predict, predict_mi_attack_dbp, compute_save_prediction_results_2, quantize_float
import train_test_helper_funcs_regression
from sklearn.externals import joblib
# import pickle

def main():
    train_pids_list, test_pids_list = get_train_test_split_2()

    regressors = joblib.load(train_test_helper_funcs_regression.regression_model_save_filename)
    # regressors = pickle.load(open(train_test_helper_funcs_regression.regression_model_save_filename, 'rb'))

    fold = 1
    index = 0
    avg_mape = 0
    mi_attack_dbp_avg_mape = 0

    for regressor in regressors:
        print('{}-fold cross validation fold {}'.format(len(regressors), fold))

        print('DEBUG - Computing and saving prediction results...')

        ini_time = time.time()

        mape, mi_attack_dbp_mape = compute_save_prediction_results_2(regressor, test_pids_list[index], fold)
        avg_mape += mape
        mi_attack_dbp_avg_mape += mi_attack_dbp_mape
        print('DEBUG - Computed and saved prediction results to ' + str(train_test_helper_funcs_regression.prediction_results_filename))
        print('DEBUG - Computed and saved MI attack DBP prediction results to ' + train_test_helper_funcs_regression.mi_attack_dbp_prediction_results_filename)

        print('INFO - Execution time for computing and saving prediction results: ' + str(quantize_float(time.time() - ini_time)) + ' s')

        print('INFO - Mean Absolute percentage error (MAPE) = ' + str(mape))
        print('INFO - MI Attack DBP Mean Absolute percentage error (MAPE) = ' + str(mi_attack_dbp_mape))

        fold += 1
        index += 1

    avg_mape /= len(regressors)
    print('INFO - Average MAPE = ' + str(quantize_float(avg_mape)))
    mi_attack_dbp_avg_mape /= len(regressors)
    print('INFO - MI Attack DBP Average MAPE = ' + str(quantize_float(mi_attack_dbp_avg_mape)))

    # print('DEBUG - Getting test data...')
    # x_test, y_expect = get_test_data(test_pids_list[index])
    # print('DEBUG - Got test data...')

    # print('Actual SBP = ' + str(y_expect))
    # y_pred = predict(regressor, x_test)
    # print('Predicted SBP = ' + str(y_pred))
    # print('INFO - Absolute Percentage error = ' + str(quantize_float(abs(y_pred - y_expect) / y_expect * 100)))

if __name__ == '__main__':
    main()