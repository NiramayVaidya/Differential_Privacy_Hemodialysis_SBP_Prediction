import sys
import numpy as np
from add_noise_vip_sbp import *
from predict_helper_funcs import predict, compute_save_prediction_results, quantize_float
import time as t

if __name__ == '__main__':
    pids = []
    with open('idp_cleaned.csv', 'r') as idp:
        line = idp.readline()
        line = idp.readline()
        while line is not '':
            line = line.strip().split(',')
            pids.append(int(line[0]))
            print(line[0], end=' ')
            line = idp.readline()
    print('\n')
    pid = int(input('Enter pid by selecting one from the above: '))
    while pid not in pids:
        pid = int(input('Enter pid by selecting one from the above: '))
    print('\n')
    dates = []
    found = False
    with open('d1_cleaned.csv', 'r') as d1:
        line = d1.readline()
        line = d1.readline()
        while line is not '' and not found:
            line = line.strip().split(',')
            while int(line[0]) == pid:
                found = True
                dates.append(line[1])
                print(line[1], end=' ')
                line = d1.readline().strip().split(',')
                if line[0] is '':
                    break
            line = d1.readline()
    print('\n')
    if dates == [] or dates[0] is '':
        print('Rerun the code and select another pid, this pid does not have any associated dates')
        sys.exit(0)
    date = input('Enter date by selecting one from the above: ')
    while date not in dates:
        date = input('Enter date by selecting one from the above: ')
    print('\n')
    times = []
    sbp_values = []
    noised_sbp_values = []
    dbp_values = []
    found = False
    with open('vip_cleaned.csv', 'r') as vip, open('vip_cleaned_sbp_noised.csv', 'r') as vip_noised:
        line_v = vip.readline()
        line_v = vip.readline()
        line_vp = vip_noised.readline()
        line_vp = vip_noised.readline()
        while line_v is not '' and not found:
            line_v = line_v.strip().split(',')
            line_vp = line_vp.strip().split(',')
            while line_v[0] + ' ' + line_v[1] == str(pid) + ' ' + date:
                found = True
                times.append(int(line_v[-1]))
                sbp_values.append(int(line_v[3]))
                noised_sbp_values.append(int(line_vp[3]))
                dbp_values.append(int(line_v[4]))
                print(line_v[-1], end=' ')
                line_v = vip.readline().strip().split(',')
                line_vp = vip_noised.readline().strip().split(',')
                if line_v[0] is '':
                    break
            line_v = vip.readline()
            line_vp = vip_noised.readline()
    print('\n')
    if times == []:
        print('Rerun the code and select another date or another pid if this is the only listed date, this date does not have any associated times')
        sys.exit(0)
    elif len(times) == 1:
        print('Rerun the code and select another date, this date has only one associated time')
        sys.exit(0)
    '''
    # Since linear regression needs a minimum of 3 data points to find best fit line, the user will not be able to predict SBP
    # for the first 3 times
    time = int(input('Enter time by selecting one from the above except the last and the first 2: '))
    while time not in times or time == times[-1] or time in times[:2]:
        time = int(input('Enter time by selecting one from the above except the last and the first 2: '))
    print('\n')
    print('For the next measure time ' + str(times[times.index(time) + 1]) + ' ->')
    print('True SBP = ' + str(sbp_values[times.index(time) + 1]))
    x_train = np.array([[dbp_value, time] for dbp_value, time in \
        zip(dbp_values[:times.index(time) + 1], times[:times.index(time) + 1])])
    y_train = np.array(sbp_values[:times.index(time) + 1]).reshape(-1, 1)
    print('Epsilon unused, no noise addition')
    print('Predicted SBP without noise = ' + str(predict(x_train, y_train, [dbp_values[times.index(time) + 1], times[times.index(time) + 1]])))
    y_train = np.array(noised_sbp_values[:times.index(time) + 1]).reshape(-1, 1)
    print('Epsilon = ' + str(epsilon))
    print('Predicted SBP with noise = ' + str(predict(x_train, y_train, [dbp_values[times.index(time) + 1], times[times.index(time) + 1]])))
    '''
    time = int(input('Enter time by selecting one from the above except the first 3: '))
    while time not in times or time in times[:3]:
        time = int(input('Enter time by selecting one from the above except the first 3: '))
    print('\n')
    actual_sbp = sbp_values[times.index(time)]
    print('Actual SBP = ' + str(actual_sbp))
    x_train = np.array([[dbp_value, time] for dbp_value, time in zip(dbp_values[:times.index(time)], times[:times.index(time)])])
    y_train = np.array(sbp_values[:times.index(time)]).reshape(-1, 1)
    predicted_sbp = predict(x_train, y_train, [dbp_values[times.index(time)], times[times.index(time)]], True)
    print('Predicted SBP = ' + str(predicted_sbp))
    print('INFO - Absolute Percentage error = ' + str(quantize_float(abs(predicted_sbp - actual_sbp) / actual_sbp * 100)))

    print('DEBUG - Computing and saving prediction results...')

    ini_time = t.time()

    mape = compute_save_prediction_results()
    print('DEBUG - Computed and saved prediction results to prediction_results_regression.txt')

    print('INFO - Execution time for computing and saving prediction results: ' + str(quantize_float(t.time() - ini_time)) + ' s')

    print('INFO - Mean Absolute percentage error (MAPE) = ' + str(mape))