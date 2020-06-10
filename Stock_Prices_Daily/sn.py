# Import intrinio
import intrinio_sdk as intrin
from intrinio_sdk.rest import ApiException
#Accesses the API with Sandbox Key
intrin.ApiClient().configuration.api_key['api_key'] = 'OmM0YjJhNTNiNmVlMGYxMGJjODIwNWIyMTU3NGJjMzgw'
#This file will keep track of all the moving variables and we can slowly add to that file
import configKeys
# Get the data of the stock AAPL
#data = yf.download('AAPL','2016-01-01','2018-01-01')
# Plot the close price of the AAPL
#data.Close.plot()
#plt.show()
import datetime
from datetime import timedelta
import pandas as pd
import collections
import copy
import os
import csv
import math
import time

from pathlib import Path

#intrinDF is a dataframe reading in all the info from companylist.csv
#the skiprows avoids the second line of the csv
intrinDF = pd.read_csv("companylist.csv", low_memory=False, skiprows = range(1,1))

#intrinTickers holds all the tickers from the companylist.csv
intrinTickers = intrinDF["Symbol"]

#.SecurityApi() is used to declare we are searching for stocks (Alternate options are for example OptionsApi, MunicipalityApi)
security_api = intrin.SecurityApi()

firstIndex = datetime.datetime.strptime(configKeys.STARTPULL, '%Y-%m-%d') + timedelta(days=1) # date | Return prices on or after the date (optional)
lastIndex = datetime.datetime.strptime(configKeys.ENDPULL, '%Y-%m-%d') - timedelta(days=1) # date | Return prices on or before the date (optional)

#This calculates the number of days needed to look back as defined by the dates in configKeys and using that number as the page_size (number of requests)
necessaryPageSize = 0;
numDays = abs((firstIndex - lastIndex).days)
necessaryPageSize = math.ceil(numDays)

#holds the stocks that have data between our start and end dates
successfulPulls = [["Symbol", "Sector"]]

#this is necessary while using the free version of intrinio because we only have access to the DOW 30 and other international stocks
workingTickers = []
acc = 0
for ourTicker in intrinTickers:
    if acc == 100:
        time.sleep(60)
        acc = 0
    try:
        stockInfo = security_api.get_security_by_id(ourTicker)
    except:
        continue

    if stockInfo.first_stock_price <= firstIndex.date():
        workingTickers.append(ourTicker)

for intrinTicker in workingTickers:
    print(intrinTicker)
    #sleep takes in an integer representing the number of seconds to pause (we need to pause because of our limited access to API calls)
    time.sleep(60)
    tickerRowIndex = 0
    #Result =  boolean dataframe with True at the position where the the intrinTicker is in intrinDF
    result = intrinDF.isin([intrinTicker])
    #By fetching the index of the row where the boolean dataframe has the value True
    tickerRowIndex = result["Symbol"][result["Symbol"] == True].index

    #initialization of a dataframe to help us keep track of stock price data over time for each security
    stockPriceDF = []
    try:
        #stockDataSummary is a list of StockPriceSummary objects, documentation for it can be found here: https://docs.intrinio.com/documentation/python/get_security_stock_prices_v2
        #NOTE: The date will always represent the last day in the period (E.g. End of the week, End of the day)
        stockDataSummary = security_api.get_security_stock_prices(intrinTicker, start_date=firstIndex, end_date=lastIndex, frequency="daily", page_size=necessaryPageSize)

    except ApiException as e:
        print("Exception when calling SecurityApi->get_security_stock_prices: %s\r\n" % e)
        continue

    #This checks if the data is within our timeframe, there's no need to check if data is empty because of the ApiException above
    if stockDataSummary.stock_prices[-1].date == firstIndex.date() and stockDataSummary.stock_prices[0].date == lastIndex.date():
        successfulPulls.append([intrinDF["Symbol"][tickerRowIndex], intrinDF["Sector"][tickerRowIndex]])
    else:
        continue

    #accessing one of our stock prices to grab the correct keys for data collection (pop the last because it is a value called "discriminator")
    unformatCols = list(stockDataSummary.stock_prices[0].__dict__.keys())
    unformatCols.pop()
    formatCols = []
    #Formats the columns to be capitalized and omit their first character (which will always be an underscore)
    for columnName in unformatCols:
        colName = columnName[1:]
        colName = colName.capitalize()
        formatCols.append(colName)

    if len(stockDataSummary.stock_prices) > 0:
        stockDSList = [] #stockDSList will hold tuples consisting of data corresponding to the headers of the df
        for i in range(len(stockDataSummary.stock_prices)):
            sdp = stockDataSummary.stock_prices[i] #sdp is a Stock Data Point
            stockDSList.append([sdp.date,sdp.intraperiod,sdp.frequency, sdp.open, sdp.high, sdp.low, sdp.close, sdp.volume, sdp.adj_open, sdp.adj_high, sdp.adj_low, sdp.adj_close, sdp.adj_volume])
        stockPriceDF = pd.DataFrame(stockDSList, columns = formatCols)
        stockPriceDF.to_csv(os.path.join(Path(configKeys.SN_DATA_FOLDER), intrinTicker+"_SNDaily.csv"))

#Creating a successful file that includes stock tickers and sectors
with open("successfulPullsSN.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(successfulPulls)
'''
#USE THIS CODE WHEN READING THE CSV's INTO OTHER PYTHON FILES!!!
'''
'''
masterStockList = []

stock = Stock(df['Symbol'][ind], stockData, df['Sector'][ind], df['IPOyear'][ind], configKeys.STARTPULL+configKeys.ENDPULL)
masterStockList.append(stock)

for i in masterStockList:
    print(i.symbol)
'''
