# vip_cleaned_.csv
# vip_cleaned_for_time_perturbed_.csv

if __name__ == '__main__':
    with open('vip_files/vip_sorted_.csv', 'r') as vip:
        with open('vip_files/vip_cleaned_for_time_perturbed_.csv', 'w') as vip_cleaned:
            time = -1
            vip_cleaned.write(vip.readline().strip() + '\n')
            line = vip.readline()
            while line is not '':
                line = line.strip().split(',')
                # if int(line[-1]) != time and int(line[-1]) != 0:
                #     time = int(line[-1])
                #     line[1] = line[1][:-9]
                #     vip_cleaned.write(','.join(line) + '\n')
                line[1] = line[1][:-9]
                vip_cleaned.write(','.join(line) + '\n')
                line = vip.readline()