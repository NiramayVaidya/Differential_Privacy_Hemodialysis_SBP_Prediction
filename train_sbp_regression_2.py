import time
from train_test_helper_funcs_regression import get_train_test_split_2, get_training_data, train, quantize_float
from sklearn.externals import joblib
# import pickle

def main():
    train_pids_list, test_pids_list = get_train_test_split_2()

    regressors = []

    fold = 1

    for train_pids in train_pids_list:
        print('{}-fold cross validation fold {}'.format(len(train_pids_list), fold))

        print('DEBUG - Getting training data...')
    
        ini_time = time.time()

        x_train, y_train = get_training_data(train_pids)
    
        print('DEBUG - Got training data')

        print('INFO - Execution time for getting training data: ' + str(quantize_float(time.time() - ini_time)) + ' s')

        regressor = train(x_train, y_train)
        regressors.append(regressor)

        fold += 1

    joblib.dump(regressors, 'regression_2.sav')
    # pickle.dump(regressors, open('regression_2.sav', 'wb'))

if __name__ == '__main__':
    main()