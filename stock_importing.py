# Import the plotting library
import matplotlib.pyplot as plt

# Import the yfinance. If you get module not found error the run !pip install yfiannce from your Jupyter notebook
import yfinance as yf

# Get the data of the stock AAPL
#data = yf.download('AAPL','2016-01-01','2018-01-01')

# Plot the close price of the AAPL
#data.Close.plot()
#plt.show()

import pandas as pd
import collections
import copy

Stock = collections.namedtuple('Stock', ['symbol', 'price', 'sector', 'IPOyear', "dates"])

df = pd.read_csv("companylist.csv", low_memory=False)

stockTickers = df.Symbol

masterStockList = []
startPull = '2016-01-01'
endPull = '2019-01-01'
csvIndex = 1
for stockName in stockTickers:
    stockData = yf.download(stockName, startPull, endPull)
    stock = Stock(stockName, stockData, df.loc[csvIndex, ['Sector']], df.loc[csvIndex, ['IPOyear']], startPull+endPull)
    masterStockList.append(stock)

for i in masterStockList:
    print(i.symbol)
