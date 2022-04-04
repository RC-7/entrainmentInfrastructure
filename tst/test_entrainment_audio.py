import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

headers = ['frequency1', 'frequency2']

df = pd.read_csv('test.csv', names=headers)

df.plot()

plt.show()