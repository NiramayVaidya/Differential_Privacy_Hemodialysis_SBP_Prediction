import sys
import numpy as np

'''
With DP
Hold out and cross validation
vip_cleaned_dp_0_1.csv
vip_cleaned_dp_1.csv
vip_cleaned_dp_2.csv
'''
vip_filename = 'vip_files/vip_cleaned_dp_2.csv'

'''
200, 250
'''
sensitivity = 200
'''
0.1, 1, 2
'''
epsilon = 2

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 add_noise_vip_sbp.py <noise_type>')
        print('Noise type: laplace/gaussian')
        sys.exit(0)
    with open('vip_files/vip_cleaned.csv', 'r') as vip_cleaned:
        # with open('vip_cleaned_sbp_noised.csv', 'w') as vip_cleaned_sbp_noised:
        with open(vip_filename, 'w') as vip_cleaned_sbp_noised:
            vip_cleaned_sbp_noised.write(vip_cleaned.readline().strip() + '\n')
            line = vip_cleaned.readline()
            while line is not '':
                line = line.strip().split(',')
                if sys.argv[1] == 'laplace':
                    # line[3] = str(int(line[3]) + int(np.random.laplace(0, sensitivity / epsilon, (1, ))[0]))
                    # while int(line[3]) == 0:
                    #     line[3] = str(int(line[3]) + int(np.random.laplace(0, sensitivity / epsilon, (1, ))[0]))
                    # line[4] = str(int(line[4]) + int(np.random.laplace(0, sensitivity / epsilon, (1, ))[0]))
                    # line[-1] = str(int(line[-1]) + int(np.random.laplace(0, sensitivity / epsilon, (1, ))[0]))
                    line[3] = str(float(line[3]) + np.random.laplace(0, sensitivity / epsilon, (1, ))[0])
                    line[4] = str(float(line[4]) + np.random.laplace(0, sensitivity / epsilon, (1, ))[0])
                    # line[-1] = str(float(line[-1]) + np.random.laplace(0, sensitivity / epsilon, (1, ))[0])
                elif sys.argv[1] == 'gaussian':
                    # line[3] = str(int(line[3]) + int(np.random.normal(0, sensitivity / epsilon, (1, ))[0]))
                    # while int(line[3]) == 0:
                    #     line[3] = str(int(line[3]) + int(np.random.laplace(0, sensitivity / epsilon, (1, ))[0]))
                    # line[4] = str(int(line[4]) + int(np.random.normal(0, sensitivity / epsilon, (1, ))[0]))
                    # line[-1] = str(int(line[-1]) + int(np.random.normal(0, sensitivity / epsilon, (1, ))[0]))
                    line[3] = str(float(line[3]) + np.random.normal(0, sensitivity / epsilon, (1, ))[0])
                    line[4] = str(float(line[4]) + np.random.normal(0, sensitivity / epsilon, (1, ))[0])
                    # line[-1] = str(float(line[-1]) + np.random.normal(0, sensitivity / epsilon, (1, ))[0])
                else:
                    print('Incorrect option provided for noise type')
                    sys.exit(0)
                vip_cleaned_sbp_noised.write(','.join(line) + '\n')
                line = vip_cleaned.readline()