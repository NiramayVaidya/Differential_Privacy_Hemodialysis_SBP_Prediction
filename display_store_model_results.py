import pandas as pd
from pandas import ExcelWriter
import numpy as np

def main():
    with open('model_results.txt', 'r') as results:
        line = results.readline()
        model_results = results.readline().strip().split(', ')
    excel_writer = ExcelWriter('model_results.xlsx', engine='xlsxwriter', options={'strings_to_numbers': False})
    models = pd.MultiIndex.from_product([['Regression', 'Neural Network', 'Neural Network Tensorflow'], ['Hold Out', '5-Fold Cross Validation']])
    df = pd.DataFrame(np.array([model_results]), columns=models)
    print(df)
    df.to_excel(excel_writer, 'Results')
    excel_writer.save()

if __name__ == '__main__':
    main()