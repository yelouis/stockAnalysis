# Import the plotting library
import matplotlib.pyplot as plt

# Import the yfinance. If you get module not found error the run !pip install yfiannce from your Jupyter notebook
import yfinance as yf

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

Stock = collections.namedtuple('Stock', ['symbol', 'price', 'sector', 'IPOyear', "dates"])

df = pd.read_csv("companylist.csv", low_memory=False)

stockTickers = df.Symbol

masterStockList = []
# csvIndex = 1
# for stockName in stockTickers:
#     stockData = yf.download(stockName, configKeys.STARTPULL, configKeys.ENDPULL)
#     stock = Stock(stockName, stockData, df.loc[csvIndex, ['Sector']], df.loc[csvIndex, ['IPOyear']], configKeys.STARTPULL+configKeys.ENDPULL)
#     masterStockList.append(stock)


for ind in df.index:
    if(df['Sector'][ind] == configKeys.SECTORLIST):
        stockData = yf.download(df['Symbol'][ind], configKeys.STARTPULL, configKeys.ENDPULL)
        stock = Stock(df['Symbol'][ind], stockData, df['Sector'][ind], df['IPOyear'][ind], configKeys.STARTPULL+configKeys.ENDPULL)
        masterStockList.append(stock)





for i in masterStockList:
    print(i.symbol)
