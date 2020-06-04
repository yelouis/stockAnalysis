# Import the plotting library
import matplotlib.pyplot as plt
# Import intrinio
import intrinio_sdk as intrin
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


from pathlib import Path

Stock = collections.namedtuple('Stock', ['symbol', 'price', 'sector', 'IPOyear', "dates"])

#df = pd.read_csv("companylist.csv", low_memory=False)
#This reads one stock into a dataframe which is : American Express for testing purposes
intrinDF = pd.read_csv("companylist.csv", low_memory=False, header = 0, skiprows = range(1, 267), nrows = 2)
'''
print("Size")
print(intrinDF.size) #14 = Represents Header Values, and ACB row values
print("Indeces")
print(intrinDF.index) # RangeIndex(start=0, stop=2, step=1)
print("Columns")
print(intrinDF.columns) # Index(['Symbol', 'Name', 'Unnamed: 2', 'Unnamed: 3', 'IPOyear', 'Sector', 'industry'], dtype='object')
'''

#This should be ACB
intrinTicker = intrinDF.at[1, "Symbol"]
if intrinTicker != "AXP":
    print("Ticker set incorrectly")

#.SecurityApi() is used to declare we are searching for stocks (Alternate options are for example OptionsApi, MunicipalityApi)
security_api = intrin.SecurityApi()

firstIndex = datetime.datetime.strptime(configKeys.STARTPULL, '%Y-%m-%d') + timedelta(days=1) # date | Return prices on or after the date (optional)
lastIndex = datetime.datetime.strptime(configKeys.ENDPULL, '%Y-%m-%d') - timedelta(days=1) # date | Return prices on or before the date (optional)

#This calculates the number of weeks needed to look back as defined by the dates in configKeys and using that number as the page_size (number of datapoints)
necessaryPageSize = 0;
numWeeks = abs((firstIndex - lastIndex).days)/7
necessaryPageSize = math.ceil(numWeeks)

print("necessaryPageSize should be 209: " + str(necessaryPageSize))

try:
    #stockDataSummary is a list of StockPriceSummary objects, documentation for it can be found here: https://docs.intrinio.com/documentation/python/get_security_stock_prices_v2
    #NOTE: The date will always represent the last day in the period (E.g. End of the week)
    stockDataSummary = security_api.get_security_stock_prices(intrinTicker, start_date=firstIndex, end_date=lastIndex, frequency="weekly", page_size=10)
    #extract data

except ApiException as e:
    print("Exception when calling SecurityApi->get_security_stock_prices: %s\r\n" % e)

sucessfulPulls = [["Symbol", "Sector"]]

#if necessaryPageSize != len(stockDataSummary.stock_prices):
#    print("you goofed")

'''
print("Stock data first date:" + stockDataSummary.stock_prices[-1].date.strftime("%m/%d/%Y"))
print("Config keys first date:" + firstIndex.strftime("%m/%d/%Y"))
print("Stock data last date:" + stockDataSummary.stock_prices[0].date.strftime("%m/%d/%Y"))
print("Config keys last date:" + lastIndex.strftime("%m/%d/%Y"))
'''
#This checks if the data is within our timeframe, there's no need to check if data is empty because of the ApiException above
if stockDataSummary.stock_prices[-1].date == firstIndex and stockDataSummary.stock_price[0].date == lastIndex:
    successfulPulls.append(intrinDF.at[1, "Symbol"], intrinDF.at[1, "Sector"])

#accessing one of our stock prices to grab the correct keys for data collection (pop the last because it is a value called "discriminator")
unformatCols = list(stockDataSummary.stock_prices[0].__dict__.keys())
unformatCols.pop()
formatCols = []
#Formats the columns to be capitalized and omit their first character (which will always be an underscore)
for columnName in unformatCols:
    colName = columnName[1:]
    colName = colName.capitalize()
    formatCols.append(colName)

#print ("Column Names: " + str(formatCols))

stockPriceDF = []
if len(stockDataSummary.stock_prices) > 0:
    stockDSList = [] #stockDSList will hold tuples consisting of data corresponding to the headers of the df
    for i in range(len(stockDataSummary.stock_prices)):
        sdp = stockDataSummary.stock_prices[i] #sdp is a Stock Data Point
        stockDSList.append([sdp.date,sdp.intraperiod,sdp.frequency, sdp.open, sdp.high, sdp.low, sdp.close, sdp.volume, sdp.adj_open, sdp.adj_high, sdp.adj_low, sdp.adj_close, sdp.adj_volume])
    stockPriceDF = pd.DataFrame(stockDSList, columns = formatCols)

print ("Data in row 0")
print (stockPriceDF.iloc[[0]])




'''
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
        stockData.to_csv(os.path.join(Path(configKeys.DATA_FOLDER), df['Symbol'][ind]+'Daily.csv'))
''''''
#Creating a sucessful file that includes stock tickers and sectors
with open("successfulPulls.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(sucessfulPulls)
'''
#USE THIS CODE WHEN READING THE CSV's INTO OTHER PYTHON FILES!!!
'''
masterStockList = []

stock = Stock(df['Symbol'][ind], stockData, df['Sector'][ind], df['IPOyear'][ind], configKeys.STARTPULL+configKeys.ENDPULL)
masterStockList.append(stock)

for i in masterStockList:
    print(i.symbol)
'''
