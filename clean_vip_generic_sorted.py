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
vip_filename = 'vip_files/vip_cleaned_dp_2_time_perturbed_.csv'

if __name__ == '__main__':
    with open('vip_generic_sorted.csv', 'r') as vip:
        with open(vip_filename, 'w') as vip_cleaned:
            time = -1
            vip_cleaned.write(vip.readline().strip() + '\n')
            line = vip.readline()
            while line is not '':
                line = line.strip().split(',')
                # if int(line[-1]) != time and int(line[-1]) != 0:
                if float(line[-1]) != time and float(line[-1]) != 0:
                    # time = int(line[-1])
                    time = float(line[-1])
                    line[1] = line[1][:-9]
                    vip_cleaned.write(','.join(line) + '\n')
                line = vip.readline()