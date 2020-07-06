import numpy as np
import sys
from decimal import *

'''
Naming convention
Files ending without _ are my files
Files ending with _ are Sneh's files
'''

'''
Without DP
Hold out and cross validation
vip_cleaned.csv
vip_cleaned_.csv

With DP
Hold out and cross validation
vip_cleaned_dp_0_1.csv
vip_cleaned_dp_1.csv
vip_cleaned_dp_2.csv
vip_cleaned_dp_0_1_.csv
vip_cleaned_dp_1_.csv
vip_cleaned_dp_2_.csv
'''
vip_filename = 'vip_files/vip_cleaned_dp_2_.csv'

vip_const_filename = 'vip_files/vip_cleaned_.csv'

d1_filename = 'd1_cleaned.csv'
idp_filename = 'idp_cleaned.csv'

'''
Without DP
Hold out
prediction_results.txt
prediction_results_.txt
Cross validation
prediction_results_2.txt
prediction_results_2_.txt

With DP (0.1, 1, 2)
Hold out
prediction_results_dp_0_1.txt
prediction_results_dp_1.txt
prediction_results_dp_2.txt
prediction_results_dp_0_1_.txt
prediction_results_dp_1_.txt
prediction_results_dp_2_.txt
Cross validation
prediction_results_2_dp_0_1.txt
prediction_results_2_dp_1.txt
prediction_results_2_dp_2.txt
prediction_results_2_dp_0_1_.txt
prediction_results_2_dp_1_.txt
prediction_results_2_dp_2_.txt
'''
prediction_results_filename = 'results_neural_network/prediction_results_dp_2_.txt'

'''
Without DP
Hold out
network_params.json
network_params_.json
Cross validation
network_params_2.json
network_params_2_.json

With DP (0.1, 1, 2)
Hold out
neural_network_dp_0_1.json
neural_network_dp_1.json
neural_network_dp_2.json
neural_network_dp_0_1_.json
neural_network_dp_1_.json
neural_network_dp_2_.json
Cross validation
neural_network_2_dp_0_1.json
neural_network_2_dp_1.json
neural_network_2_dp_2.json
neural_network_2_dp_0_1_.json
neural_network_2_dp_1_.json
neural_network_2_dp_2_.json
'''
neural_network_model_save_filename = 'saved_models_neural_network/neural_network_dp_2_.json'

def quantize_float(num):
    return float(Decimal(num).quantize(Decimal('1.00')))

def get_train_test_split():
    pids = []
    with open(idp_filename, 'r') as idp:
        line = idp.readline()
        line = idp.readline()
        while line is not '':
            line = line.strip().split(',')
            pids.append(int(line[0]))
            line = idp.readline()
    train_test_split_point = int(len(pids) * 0.8)
    train_pids = pids[:train_test_split_point]
    test_pids = pids[train_test_split_point:]
    return (train_pids, test_pids)

def get_train_test_split_2():
    pids = []
    train_pids_list = []
    test_pids_list = []
    with open(idp_filename, 'r') as idp:
        line = idp.readline()
        line = idp.readline()
        while line is not '':
            line = line.strip().split(',')
            pids.append(int(line[0]))
            line = idp.readline()
    train_test_split_point = int(len(pids) * 0.8)
    train_pids_list.append(pids[:train_test_split_point])
    test_pids_list.append(pids[train_test_split_point:])

    for iterator in range(3, 0, -1):
        train_test_split_point_1 = int(len(pids) * 0.2 * iterator)
        train_test_split_point_2 = int(len(pids) * 0.2 * (iterator + 1))
        train_pids_list.append(pids[:train_test_split_point_1] + pids[train_test_split_point_2:])
        test_pids_list.append(pids[train_test_split_point_1:train_test_split_point_2])

    train_test_split_point = int(len(pids) * 0.2)
    train_pids_list.append(pids[train_test_split_point:])
    test_pids_list.append(pids[:train_test_split_point])

    return (train_pids_list, test_pids_list)

def vectorized_result(j):
    e = np.zeros((250, 1))
    e[int(j)] = 1.0
    return e

def get_training_data(train_pids):
    train_sbp_values = []
    train_dbp_values = []
    train_times = []
    with open(vip_filename, 'r') as vip, open(vip_const_filename, 'r') as vip_other:
        line = vip.readline()
        line = vip.readline()
        line_other = vip_other.readline()
        line_other = vip_other.readline()
        while line is not '' or line_other is not '':
            line = line.strip().split(',')
            line_other = line_other.strip().split(',')
            if int(line[0]) in train_pids:
                # train_sbp_values.append(int(line[3]))
                # train_sbp_values.append(float(line[3]))
                train_sbp_values.append(float(line_other[3]))
                # train_dbp_values.append(int(line[4]))
                train_dbp_values.append(float(line[4]))
                # train_times.append(int(line[-1]))
                train_times.append(float(line[-1]))
            line = vip.readline()
            line_other = vip_other.readline()
    training_inputs = [np.array(np.reshape([dbp, time], (2, 1)), dtype=np.float64) \
        for dbp, time in zip(train_dbp_values, train_times)]
    training_results = [vectorized_result(sbp) for sbp in train_sbp_values]
    return zip(training_inputs, training_results)

def get_test_data(test_pids):
    for pid in test_pids:
        print(pid, end=' ')
    print('\n')
    pid = int(input('Enter pid by selecting one from the above: '))
    while pid not in test_pids:
        pid = int(input('Enter pid by selecting one from the above: '))
    print('\n')

    dates = []
    found = False
    with open(d1_filename, 'r') as d1:
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
    if dates == []:
        print('Rerun the code and select another pid, this pid does not have any associated dates')
        sys.exit(0)
    date = input('Enter date by selecting one from the above: ')
    while date not in dates:
        date = input('Enter date by selecting one from the above: ')
    print('\n')

    times = []
    sbp_values = []
    dbp_values = []
    found = False
    with open(vip_const_filename, 'r') as vip:
        line = vip.readline()
        line = vip.readline()
        while line is not '' and not found:
            line = line.strip().split(',')
            while line[0] + ' ' + line[1] == str(pid) + ' ' + date:
                found = True
                # times.append(int(line[-1]))
                times.append(float(line[-1]))
                # sbp_values.append(int(line[3]))
                sbp_values.append(float(line[3]))
                # dbp_values.append(int(line[4]))
                dbp_values.append(float(line[4]))
                print(line[-1], end=' ')
                line = vip.readline().strip().split(',')
                if line[0] is '':
                    break
            line = vip.readline()
    print('\n')
    if times == []:
        print('Rerun the code and select another date or another pid if this is the only listed date, this date does not have any associated times')
        sys.exit(0)
    # time = int(input('Enter time by selecting one from the above: '))
    time = float(input('Enter time by selecting one from the above: '))
    # while time not in times or time == times[-1]:
    while time not in times:
        # time = int(input('Enter time by selecting one from the above: '))
        time = float(input('Enter time by selecting one from the above: '))
    print('\n')

    test_input = np.array(np.reshape([dbp_values[times.index(time)], times[times.index(time)]], (2, 1)), dtype=np.float64)
    test_result = sbp_values[times.index(time)]

    return (test_input, test_result)

def get_prediction(network, test_input):
    return np.argmax(network.feedforward(test_input)) + 1

def compute_save_prediction_results(network, test_pids):
    pid_dates = {}
    total_error = 0
    num_test_cases = 0
    for test_pid in test_pids:
        pid_dates[test_pid] = []
    with open(d1_filename, 'r') as d1:
        line = d1.readline()
        line = d1.readline()
        while line is not '':
            line = line.strip().split(',')
            if int(line[0]) in pid_dates.keys():
                pid_dates[int(line[0])].append(line[1])
            line = d1.readline()
    with open(prediction_results_filename, 'w') as results:
        results.write('Pid Date Time Actual_SBP Predicted_SBP Absolute_Percentage_Error\n')
        with open(vip_const_filename, 'r') as vip:
            line = vip.readline()
            line = vip.readline()
            while line is not '':
                line = line.strip().split(',')
                if int(line[0]) in pid_dates.keys():
                    if line[1] in pid_dates[int(line[0])]:
                        num_test_cases += 1
                        # test_input = np.array(np.reshape([int(line[4]), int(line[-1])], (2, 1)), dtype=np.float64)
                        test_input = np.array(np.reshape([float(line[4]), float(line[-1])], (2, 1)), dtype=np.float64)
                        # test_result = int(line[3])
                        test_result = float(line[3])
                        prediction = get_prediction(network, test_input)
                        # error = quantize_float(abs(prediction - test_result) / test_result * 100)
                        error = quantize_float(abs((prediction - test_result) / test_result * 100))
                        total_error += error
                        results.write(line[0] + ' ' + line[1] + ' ' + line[-1] + ' ' + line[3] + ' ' + str(prediction) + ' ' + str(error) + '\n')
                line = vip.readline()
    return quantize_float(total_error / num_test_cases)

def compute_save_prediction_results_2(network, test_pids, fold):
    pid_dates = {}
    total_error = 0
    num_test_cases = 0
    for test_pid in test_pids:
        pid_dates[test_pid] = []
    with open(d1_filename, 'r') as d1:
        line = d1.readline()
        line = d1.readline()
        while line is not '':
            line = line.strip().split(',')
            if int(line[0]) in pid_dates.keys():
                pid_dates[int(line[0])].append(line[1])
            line = d1.readline()
    mode = 'a'
    if fold == 1:
        mode = 'w'
    with open(prediction_results_filename, mode) as results:
        results.write('Fold ' + str(fold) + '\n')
        results.write('Pid Date Time Actual_SBP Predicted_SBP Absolute_Percentage_Error\n')
        with open(vip_const_filename, 'r') as vip:
            line = vip.readline()
            line = vip.readline()
            while line is not '':
                line = line.strip().split(',')
                if int(line[0]) in pid_dates.keys():
                    if line[1] in pid_dates[int(line[0])]:
                        num_test_cases += 1
                        # test_input = np.array(np.reshape([int(line[4]), int(line[-1])], (2, 1)), dtype=np.float64)
                        test_input = np.array(np.reshape([float(line[4]), float(line[-1])], (2, 1)), dtype=np.float64)
                        # test_result = int(line[3])
                        test_result = float(line[3])
                        prediction = get_prediction(network, test_input)
                        # error = quantize_float(abs(prediction - test_result) / test_result * 100)
                        error = quantize_float(abs((prediction - test_result) / test_result * 100))
                        total_error += error
                        results.write(line[0] + ' ' + line[1] + ' ' + line[-1] + ' ' + line[3] + ' ' + str(prediction) + ' ' + str(error) + '\n')
                line = vip.readline()
        results.write('\n')
    return quantize_float(total_error / num_test_cases)

if __name__ == '__main__':
    train_pids, test_pids = get_train_test_split()
    print('Pids for training: ' + str(train_pids))
    print('Pids for testing: ' + str(test_pids))

    training_data = list(get_training_data(train_pids))
    print('Training data: ' + str(training_data))

    test_input, test_result = get_test_data(test_pids)
    print('Test input: ' + str(test_input))
    print('Test result: ' + str(test_result))