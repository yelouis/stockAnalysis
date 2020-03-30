# Import the plotting library
import matplotlib.pyplot as plt

#This file will keep track of all the moving variables and we can slowly add to that file
import configKeys

# Get the data of the stock AAPL
#data = yf.download('AAPL','2016-01-01','2018-01-01')

# Plot the close price of the AAPL
#data.Close.plot()
#plt.show()

import pandas as pd
import collections
import copy
import os
import csv

from pathlib import Path

df = pd.read_csv("successfulPulls.csv", low_memory=False)

allStock = {}

for ind in df.index:
    stockDF = pd.read_csv(os.path.join(Path(configKeys.DATA_FOLDER), df['Symbol'][ind]+'Daily.csv'), low_memory=False)
    allStock[df['Symbol'][ind]] = stockDF

print(allStock)
