import time
from train_test_helper_funcs_regression import get_train_test_split, get_training_data, train, quantize_float
import train_test_helper_funcs_regression
from sklearn.externals import joblib
# import pickle

def main():
    train_pids, test_pids = get_train_test_split()

    print('DEBUG - Getting training data...')
    
    ini_time = time.time()

    x_train, y_train = get_training_data(train_pids)
    
    print('DEBUG - Got training data')

    print('INFO - Execution time for getting training data: ' + str(quantize_float(time.time() - ini_time)) + ' s')

    regressor = train(x_train, y_train)
    joblib.dump(regressor, train_test_helper_funcs_regression.regression_model_save_filename)
    # pickle.dump(regressor, open(train_test_helper_funcs_regression.regression_model_save_filename, 'wb'))

if __name__ == '__main__':
    main()