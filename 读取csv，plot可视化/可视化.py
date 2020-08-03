import numpy as np
import matplotlib.pyplot as plt



# import csv
# import pandas as pd
# sFileName='hist17.csv'
# d = pd.read_csv('hist17.csv', usecols=['accuracy'])
# df = d.to_array()
# print(type(df))
# # print(d)
# # for i in d:
# #     print(i)
import csv
import pandas as pd
with open('new_03_hist_60.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    column1 = [row[7]for row in reader]
    # print(column1[1:])
    a = column1[1:]
    # max = np.argmax(a)
    # print(max)
    # print(a[max])
    X = []
    Y = []
    for i,c1 in enumerate(column1[1:]):
        # x = np.linspace(0, 2 * np.pi, 100)
        X.append(i)

        Y.append(float(c1))

    plt.plot(X[0:60:5],Y[0:60:5],'ro-')
    plt.xlabel('epochs')
    plt.ylabel('getPrecision')
    plt.show()
