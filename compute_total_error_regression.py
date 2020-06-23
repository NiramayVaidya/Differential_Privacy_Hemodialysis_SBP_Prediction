if __name__ == '__main__':
    total_error = 0
    with open('prediction_results_regression.txt', 'r') as results:
        line = results.readline()
        line = results.readline()
        while line is not '':
            line = line.strip().split(' ')
            total_error += float(line[-1])
            line = results.readline()
    print('Total error = ' + str(total_error))
