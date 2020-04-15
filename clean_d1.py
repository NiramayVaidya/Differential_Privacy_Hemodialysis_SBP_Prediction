if __name__ == '__main__':
    with open('d1.csv', 'r') as d1:
        with open('d1_cleaned.csv', 'w') as d1_cleaned:
            d1_cleaned.write(d1.readline().strip() + '\n')
            line = d1.readline()
            while line is not '':
                line = line.strip().split(',')
                if line[-1] != 'NA':
                    line[0] = line[0][1:-1]
                    line[1] = line[1][:-9]
                    line[2] = line[2][1:-1]
                    line[3] = line[3][1:-1]
                    d1_cleaned.write(','.join(line) + '\n')
                line = d1.readline()