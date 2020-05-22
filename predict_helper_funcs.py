import numpy as np
from sklearn.linear_model import LinearRegression
from decimal import *

def quantize_float(num):
    return float(Decimal(num).quantize(Decimal('1.00')))

def predict(x_train, y_train, x_test, print_info):
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    if print_info is True:
        print('Regressor parameters: intercept = ' + str(quantize_float(regressor.intercept_[0])) + ', coefficients = ' + str(quantize_float(regressor.coef_[0][0])) + ', ' + str(quantize_float(regressor.coef_[0][1])))
        print('Regressor equation: SBP = (' + str(quantize_float(regressor.coef_[0][0])) + ' * DBP) + (' + str(quantize_float(regressor.coef_[0][1])) + ' * time_value) + ' + str(quantize_float(regressor.intercept_[0])))
    y_pred = regressor.predict(np.array([x_test]))
    return quantize_float(y_pred[0][0])

def compute_save_prediction_results():
    total_error = 0
    num_test_cases = 0
    times = []
    sbp = []
    dbp = []
    with open('prediction_results_regression.txt', 'w') as results:
        results.write('Pid Date Time Actual_SBP Predicted_SBP Absolute_Percentage_Error\n')
        with open('vip_cleaned.csv', 'r') as vip:
            line = vip.readline()
            line = vip.readline()
            line = line.strip().split(',')
            pid = int(line[0])
            date = line[1]
            times.append(int(line[-1]))
            sbp.append(int(line[3]))
            dbp.append(int(line[4]))
            line = vip.readline()
            while line is not '':
                line = line.strip().split(',')
                if pid == int(line[0]) and date == line[1]:
                    times.append(int(line[-1]))
                    sbp.append(int(line[3]))
                    dbp.append(int(line[4]))
                else:
                    if len(times) > 3:
                        for iter in range(3, len(times)):
                            num_test_cases += 1
                            actual_sbp = sbp[iter]
                            x_train = np.array([[dbp_value, time] for dbp_value, time in zip(dbp[:iter], times[:iter])])
                            y_train = np.array(sbp[:iter]).reshape(-1, 1)
                            predicted_sbp = predict(x_train, y_train, [dbp[iter], times[iter]], False)
                            error = quantize_float(abs(predicted_sbp - actual_sbp) / actual_sbp * 100)
                            total_error += error
                            results.write(str(pid) + ' ' + date + ' ' + str(times[iter]) + ' ' + str(actual_sbp) + ' ' + str(predicted_sbp) + ' ' + str(error) + '\n')
                    pid = int(line[0])
                    date = line[1]
                    times.clear()
                    sbp.clear()
                    dbp.clear()
                    times.append(int(line[-1]))
                    sbp.append(int(line[3]))
                    dbp.append(int(line[4]))
                line = vip.readline()
        if len(times) > 3:
            for iter in range(3, len(times)):
                num_test_cases += 1
                actual_sbp = sbp[iter]
                x_train = np.array([[dbp_value, time] for dbp_value, time in zip(dbp[:iter], times[:iter])])
                y_train = np.array(sbp[:iter]).reshape(-1, 1)
                predicted_sbp = predict(x_train, y_train, [dbp[iter], times[iter]], False)
                error = quantize_float(abs(predicted_sbp - actual_sbp) / actual_sbp * 100)
                total_error += error
                results.write(str(pid) + ' ' + date + ' ' + str(times[iter]) + ' ' + str(actual_sbp) + ' ' + str(predicted_sbp) + ' ' + str(error) + '\n')
    return quantize_float(total_error / num_test_cases)

if __name__ == '__main__':
    mape = compute_save_prediction_results()
    print('Mean Absolute percentage error (MAPE) = ' + str(mape))