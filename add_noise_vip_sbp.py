import sys
import numpy as np

sensitivity = 250
epsilon = 5

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 add_noise_vip_sbp.py <noise_type>')
        print('Noise type: laplace/gaussian')
        sys.exit(0)
    with open('vip_cleaned.csv', 'r') as vip_cleaned:
        with open('vip_cleaned_sbp_noised.csv', 'w') as vip_cleaned_sbp_noised:
            vip_cleaned_sbp_noised.write(vip_cleaned.readline().strip() + '\n')
            line = vip_cleaned.readline()
            while line is not '':
                line = line.strip().split(',')
                if sys.argv[1] == 'laplace':
                    line[3] = str(int(line[3]) + int(np.random.laplace(0, sensitivity / epsilon, (1, )[0])))
                elif sys.argv[1] == 'gaussian':
                    line[3] = str(int(line[3]) + int(np.random.normal(0, sensitivity / epsilon, (1, )[0])))
                else:
                    print('Incorrect option provided for noise type')
                    sys.exit(0)
                vip_cleaned_sbp_noised.write(','.join(line) + '\n')
                line = vip_cleaned.readline()