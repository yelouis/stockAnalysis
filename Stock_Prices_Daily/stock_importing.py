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
import os
import csv

from pathlib import Path

Stock = collections.namedtuple('Stock', ['symbol', 'price', 'sector', 'IPOyear', "dates"])

df = pd.read_csv("companylist.csv", low_memory=False)

stockTickers = df.Symbol

data_folder = Path("Daily/")

sucessfulPulls = [["Symbol", "Sector"]]

for ind in df.index:
    # Have an if statement in place in case if we don't want to pull every stock because there are a lot of stocks
    # Program takes a long time to run if we have to webscrape every stock each time we run
    stockData = []
    # Code to get daily data
    # stockData = yf.download(df['Symbol'][ind], start=configKeys.STARTPULL, end=configKeys.ENDPULL)

    # https://pypi.org/project/yfinance/
    stockData = yf.download(df['Symbol'][ind], start=configKeys.STARTPULL, end=configKeys.ENDPULL)

    if stockData.empty == False:
        sucessfulPulls.append([df['Symbol'][ind], df['Sector'][ind]])


    # If there's something that's been loaded into stockData, then the length is no longer 0
    if len(stockData) > 0:
        stockData = stockData.assign(Sector = df['Sector'][ind], IPOyear = df['IPOyear'][ind])
        stockData.to_csv(os.path.join(data_folder, df['Symbol'][ind]+'Daily.csv'))


#Creating a sucessful file that includes stock tickers and sectors
with open("sucessfulPulls.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(sucessfulPulls)

#USE THIS CODE WHEN READING THE CSV's INTO OTHER PYTHON FILES!!!
'''
masterStockList = []

stock = Stock(df['Symbol'][ind], stockData, df['Sector'][ind], df['IPOyear'][ind], configKeys.STARTPULL+configKeys.ENDPULL)
masterStockList.append(stock)

for i in masterStockList:
    print(i.symbol)
'''
