if __name__ == '__main__':
    with open('idp.csv', 'r') as idp:
        with open('idp_cleaned.csv', 'w') as idp_cleaned:
            idp_cleaned.write(idp.readline().strip() + '\n')
            line = idp.readline()
            while line is not '':
                line = line.strip().split(',')
                line[0] = line[0][1:-1]
                line[1] = line[1][1:-1]
                line[3] = line[3][1:-1]
                idp_cleaned.write(','.join(line) + '\n')
                line = idp.readline()