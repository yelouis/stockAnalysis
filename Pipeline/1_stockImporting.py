# Import investpy. If you get module not found error the run pip install investpy==0.9.14
import investpy
# https://pypi.org/project/investpy/
# github: https://github.com/alvarobartt/investpy

#This file will keep track of all the moving variables and we can slowly add to that file
import _configKeys

#other importing
import pandas as pd
import os
import csv
import datetime
from datetime import timedelta
from pathlib import Path
from get_all_tickers import get_tickers as gt

global sucessfulPulls
sucessfulPulls = {"Symbol" : [],
                  "Type" : []}


def import_stocks():
    #imports stocks in the US

    search_results = investpy.search_stocks(by='country', value='united states')
    list_of_stock_tickers = search_results["symbol"]

    firstIndex = datetime.datetime.strptime(_configKeys.STARTPULL, '%d/%m/%Y')
    lastIndex = datetime.datetime.strptime(_configKeys.ENDPULL, '%d/%m/%Y')

    for ticker in list_of_stock_tickers[:2500]:

        # Have an if statement in place in case if we don't want to pull every stock because there are a lot of stocks
        # Program takes a long time to run if we have to webscrape every stock each time we run
        stockData = []

        stockData = investpy.get_stock_historical_data(stock=ticker,
                                            country='united states',
                                            from_date=_configKeys.STARTPULL,
                                            to_date=_configKeys.ENDPULL)
        # If there's something that's been loaded into stockData, then the length is no longer 0
        # if the differences is under 2~3 days, then it is ok to take this data since there is still enough data in the week to be usable
        # this timedelta fixes the problem of trying to pull during a long weekend
        if stockData.empty == False and stockData.index[0] - firstIndex <= timedelta(days = 2) and lastIndex - stockData.index[-1] <= timedelta(days = 3):
            sucessfulPulls["Symbol"].append(ticker)
            sucessfulPulls["Type"].append("Stock")
            stockData.to_csv(os.path.join(Path(_configKeys.DATA_FOLDER), ticker+'.csv'))


def import_funds():
    # imports funds in the US

    search_results = investpy.search_funds(by='country', value='united states')
    list_of_fund_names = search_results["name"]

    firstIndex = datetime.datetime.strptime(_configKeys.STARTPULL, '%d/%m/%Y')
    lastIndex = datetime.datetime.strptime(_configKeys.ENDPULL, '%d/%m/%Y')

    for name in list_of_fund_names[:2500]:

        # Have an if statement in place in case if we don't want to pull every fund because there are a lot of funds
        # Program takes a long time to run if we have to webscrape every fund each time we run
        fundData = []

        fundData = investpy.get_fund_historical_data(fund=name,
                                            country='united states',
                                            from_date=_configKeys.STARTPULL,
                                            to_date=_configKeys.ENDPULL)
        # If there's something that's been loaded into stockData, then the length is no longer 0
        # if the differences is under 2~3 days, then it is ok to take this data since there is still enough data in the week to be usable
        # this timedelta fixes the problem of trying to pull during a long weekend
        if fundData.empty == False and fundData.index[0] - firstIndex <= timedelta(days = 2) and lastIndex - fundData.index[-1] <= timedelta(days = 3):
            sucessfulPulls["Symbol"].append(name)
            sucessfulPulls["Type"].append("Fund")
            fundData.to_csv(os.path.join(Path(_configKeys.DATA_FOLDER), name+'.csv'))


def import_etfs():
    # imports ETFs in the US

    search_results = investpy.search_etfs(by='country', value='united states')
    list_of_etf_names = search_results["name"]

    firstIndex = datetime.datetime.strptime(_configKeys.STARTPULL, '%d/%m/%Y')
    lastIndex = datetime.datetime.strptime(_configKeys.ENDPULL, '%d/%m/%Y')

    for name in list_of_etf_names[:2500]:

        # Have an if statement in place in case if we don't want to pull every etf because there are a lot of stocks
        # Program takes a long time to run if we have to webscrape every etf each time we run
        etfData = []

        etfData = investpy.get_etf_historical_data(etf=name,
                                            country='united states',
                                            from_date=_configKeys.STARTPULL,
                                            to_date=_configKeys.ENDPULL)
        # If there's something that's been loaded into stockData, then the length is no longer 0
        # if the differences is under 2~3 days, then it is ok to take this data since there is still enough data in the week to be usable
        # this timedelta fixes the problem of trying to pull during a long weekend
        if etfData.empty == False and etfData.index[0] - firstIndex <= timedelta(days = 2) and lastIndex - etfData.index[-1] <= timedelta(days = 3):
            sucessfulPulls["Symbol"].append(name)
            sucessfulPulls["Type"].append("ETF")
            etfData.to_csv(os.path.join(Path(_configKeys.DATA_FOLDER), name+'.csv'))


def import_bonds():
    # imports bonds in the US

    search_results = investpy.search_bonds(by='country', value='united states')
    list_of_bond_names = search_results["name"]

    firstIndex = datetime.datetime.strptime(_configKeys.STARTPULL, '%d/%m/%Y')
    lastIndex = datetime.datetime.strptime(_configKeys.ENDPULL, '%d/%m/%Y')

    for name in list_of_bond_names[:2500]:

        # Have an if statement in place in case if we don't want to pull every etf because there are a lot of stocks
        # Program takes a long time to run if we have to webscrape every etf each time we run
        bondData = []

        bondData = investpy.get_bond_historical_data(bond=name,
                                            from_date=_configKeys.STARTPULL,
                                            to_date=_configKeys.ENDPULL)
        # If there's something that's been loaded into stockData, then the length is no longer 0
        # if the differences is under 2~3 days, then it is ok to take this data since there is still enough data in the week to be usable
        # this timedelta fixes the problem of trying to pull during a long weekend
        if (bondData.empty == False) and (bondData.index[0] - firstIndex.date() <= timedelta(days = 2)) and (lastIndex.date() - bondData.index[-1] <= timedelta(days = 3)):
            sucessfulPulls["Symbol"].append(name)
            sucessfulPulls["Type"].append("Bond")
            bondData.to_csv(os.path.join(Path(_configKeys.DATA_FOLDER), name+'.csv'))


def import_commodities():
    # imports commodities from around the world in USD

    search_results = investpy.search_commodities(by='currency', value='USD')
    list_of_commodity_names = search_results["name"]

    firstIndex = datetime.datetime.strptime(_configKeys.STARTPULL, '%d/%m/%Y')
    lastIndex = datetime.datetime.strptime(_configKeys.ENDPULL, '%d/%m/%Y')

    for name in list_of_commodity_names[:2500]:

        # Have an if statement in place in case if we don't want to pull every etf because there are a lot of stocks
        # Program takes a long time to run if we have to webscrape every etf each time we run
        commodityData = []

        commodityData = investpy.get_commodity_historical_data(commodity=name,
                                            from_date=_configKeys.STARTPULL,
                                            to_date=_configKeys.ENDPULL)
        # If there's something that's been loaded into stockData, then the length is no longer 0
        # if the differences is under 2~3 days, then it is ok to take this data since there is still enough data in the week to be usable
        # this timedelta fixes the problem of trying to pull during a long weekend
        if commodityData.empty == False and commodityData.index[0] - firstIndex <= timedelta(days = 2) and lastIndex - commodityData.index[-1] <= timedelta(days = 3):
            sucessfulPulls["Symbol"].append(name)
            sucessfulPulls["Type"].append("Commodity")
            commodityData.to_csv(os.path.join(Path(_configKeys.DATA_FOLDER), name+'.csv'))


def main():
    # This currently only gets the first 100 items from each asset type. Remove indexing if you want more data

    print("Importing Stocks")
    import_stocks()
    print("Importing Funds")
    import_funds()
    print("Importing ETFs")
    import_etfs()
    print("Importing Bonds")
    import_bonds()
    print("Importing Commodities")
    import_commodities()

    df = pd.DataFrame(sucessfulPulls, columns = ["Symbol", "Type"])

    #Creating a sucessful file that includes asset names/tickers
    df.to_csv('1successfulPulls.csv', index=False)


main()
