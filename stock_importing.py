# Import the plotting library
import matplotlib.pyplot as plt

# Import the yfinance. If you get module not found error the run !pip install yfiannce from your Jupyter notebook
import yfinance as yf

#This file will keep track of all the moving variables and we can slowly add to that file
import configKeys

<<<<<<< HEAD
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
    # Have an if statement in place in case if we don't want to pull every stock because there are a lot of stocks
    # Program takes a long time to run if we have to webscrape every stock each time we run
    if(df['Sector'][ind] == configKeys.SECTORLIST):
        # Code to get daily data
        # stockData = yf.download(df['Symbol'][ind], start=configKeys.STARTPULL, end=configKeys.ENDPULL)

        # Here is how to get hourly data. Only problem is that we can't get it over a large interval
        # https://pypi.org/project/yfinance/
        stockData = yf.download(df['Symbol'][ind], period = "ytd", interval = "1h")
        stock = Stock(df['Symbol'][ind], stockData, df['Sector'][ind], df['IPOyear'][ind], configKeys.STARTPULL+configKeys.ENDPULL)
        masterStockList.append(stock)





for i in masterStockList:
    print(i.symbol)

# Import the plotting library
import matplotlib.pyplot as plt

# Import the yfinance. If you get module not found error the run !pip install yfiannce from your Jupyter notebook
import yfinance as yf

=======
>>>>>>> 783ea5f641163f4a436272552730c10ebe3f8897
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
<<<<<<< HEAD
startPull = '2016-01-01'
endPull = '2019-01-01'
csvIndex = 0
for stockName in stockTickers:
    csvIndex += 1
    stockData = []
    stockData = yf.download(stockName, startPull, endPull)
    if len(stockData) > 1:
        stock = Stock(stockName, stockData, df.loc[csvIndex, ['Sector']], df.loc[csvIndex, ['IPOyear']], startPull+endPull)
        masterStockList.append(stock)
=======
# csvIndex = 1
# for stockName in stockTickers:
#     stockData = yf.download(stockName, configKeys.STARTPULL, configKeys.ENDPULL)
#     stock = Stock(stockName, stockData, df.loc[csvIndex, ['Sector']], df.loc[csvIndex, ['IPOyear']], configKeys.STARTPULL+configKeys.ENDPULL)
#     masterStockList.append(stock)


for ind in df.index:
    # Have an if statement in place in case if we don't want to pull every stock because there are a lot of stocks
    # Program takes a long time to run if we have to webscrape every stock each time we run
    if(df['Sector'][ind] == configKeys.SECTORLIST):
        # Code to get daily data
        # stockData = yf.download(df['Symbol'][ind], start=configKeys.STARTPULL, end=configKeys.ENDPULL)

        # Here is how to get hourly data. Only problem is that we can't get it over a large interval
        # https://pypi.org/project/yfinance/
        stockData = yf.download(df['Symbol'][ind], period = "ytd", interval = "1h")
        stock = Stock(df['Symbol'][ind], stockData, df['Sector'][ind], df['IPOyear'][ind], configKeys.STARTPULL+configKeys.ENDPULL)
        masterStockList.append(stock)




>>>>>>> 783ea5f641163f4a436272552730c10ebe3f8897

for i in masterStockList:
    print(i.symbol)
