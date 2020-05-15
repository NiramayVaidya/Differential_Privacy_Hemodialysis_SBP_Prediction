import numpy as np
import sys

def quantize_float(num):
    return float(Decimal(num).quantize(Decimal('1.00')))

def get_train_test_split():
    pids = []
    with open('idp_cleaned.csv', 'r') as idp:
        line = idp.readline()
        line = idp.readline()
        while line is not '':
            line = line.strip().split(',')
            pids.append(int(line[0]))
            line = idp.readline()
    train_test_split_point = int(len(pids) * 0.8)
    # train_pids = pids[:train_test_split_point]
    train_pids = pids[:int(len(pids) * 0.25 * 0.8)]
    # test_pids = pids[train_test_split_point:]
    test_pids = pids[int(len(pids) * 0.25 * 0.8):int(len(pids) * 0.25)]
    return (train_pids, test_pids)

def vectorized_result(j):
    e = np.zeros((250, 1))
    e[j] = 1.0
    return e

def vectorized_result_list(j):
    e = []
    for iterator in range(0, 250):
        if iterator == j - 1:
            e.append(1.0)
        else:
            e.append(0.0)
    return e

def get_training_data(train_pids):
    train_sbp_values = []
    train_dbp_values = []
    train_times = []
    with open('vip_cleaned.csv', 'r') as vip:
        line = vip.readline()
        line = vip.readline()
        while line is not '':
            line = line.strip().split(',')
            if int(line[0]) in train_pids:
                train_sbp_values.append(int(line[3]))
                train_dbp_values.append(int(line[4]))
                train_times.append(int(line[-1]))
            line = vip.readline()
    training_inputs = np.array([[1.0, dbp, time] for dbp, time in zip(train_dbp_values, train_times)], dtype=np.float64)
    training_results = np.array([vectorized_result_list(sbp) for sbp in train_sbp_values], dtype=np.float64)
    # training_results = np.array([[sbp / 250] for sbp in train_sbp_values], dtype=np.float64)
    return (training_inputs, training_results)

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
    with open('vip_cleaned.csv', 'r') as vip:
        line = vip.readline()
        line = vip.readline()
        while line is not '' and not found:
            line = line.strip().split(',')
            while line[0] + ' ' + line[1] == str(pid) + ' ' + date:
                found = True
                times.append(int(line[-1]))
                sbp_values.append(int(line[3]))
                dbp_values.append(int(line[4]))
                print(line[-1], end=' ')
                line = vip.readline().strip().split(',')
                if line[0] is '':
                    break
            line = vip.readline()
    print('\n')
    if times == []:
        print('Rerun the code and select another date or another pid if this is the only listed date, this date does not have any associated times')
        sys.exit(0)
    time = int(input('Enter time by selecting one from the above: '))
    while time not in times or time == times[-1]:
        time = int(input('Enter time by selecting one from the above: '))
    print('\n')

    test_input = np.array([[1.0, dbp_values[times.index(time)], times[times.index(time)]]], dtype=np.float64)
    # test_result = np.array([vectorized_result_list(sbp_values[times.index(time)])], dtype=np.float64)
    test_result = np.array([[sbp_values[times.index(time)]]])

    return (test_input, test_result)

def compute_save_prediction_results(test_pids, tf_session, tf_X_var, tf_y_var, tf_predict_var):
    pid_dates = {}
    for test_pid in test_pids:
        pid_dates[test_pid] = []
    with open('d1_cleaned.csv', 'r') as d1:
        line = d1.readline()
        line = d1.readline()
        while line is not '':
            line = line.strip().split(',')
            if int(line[0]) in pid_dates.keys():
                pid_dates[int(line[0])].append(line[1])
            line = d1.readline()
    with open('prediction_results.txt', 'w') as results:
        results.write('Pid Date Time Actual_SBP Predicted_SBP Percentage_error\n')
        with open('vip_cleaned.csv', 'r') as vip:
            line = vip.readline()
            line = vip.readline()
            while line is not '':
                line = line.strip().split(',')
                if int(line[0]) in pid_dates.keys():
                    if line[1] in pid_dates[int(line[0])]:
                        test_X = np.array([[1.0, int(line[4]), int(line[-1])]], dtype=np.float64)
                        # test_y = np.array([vectorized_result_list(int(line[3]))], dtype=np.float64)
                        test_y = np.array([[int(line[3])]])
                        actual_sbp = int(line[3])
                        # predicted_sbp = tf_session.run(tf_predict_var, feed_dict={tf_X_var: test_X, tf_y_var: test_y})[0] + 1
                        predicted_sbp = tf_session.run(tf_predict_var, feed_dict={tf_X_var: test_X, tf_y_var: test_y})[0] * 250
                        results.write(line[0] + ' ' + line[1] + ' ' + line[-1] + ' ' + line[3] + ' ' + str(predicted_sbp) + ' ' + str(abs(predicted_sbp - actual_sbp) / actual_sbp * 100) + '\n')
                line = vip.readline()

if __name__ == '__main__':
    train_pids, test_pids = get_train_test_split()
    print('Pids for training: ' + str(train_pids))
    print('Pids for testing: ' + str(test_pids))

    training_data = list(get_training_data())
    print('Training data: ' + str(training_data))

    test_input, test_result = get_test_data()
    print('Test input: ' + str(test_input))
    print('Test result: ' + str(test_result))