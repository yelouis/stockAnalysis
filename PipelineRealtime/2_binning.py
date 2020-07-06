#This file will keep track of all the moving variables and we can slowly add to that file
import _configKeys

import pandas as pd
import collections
import copy
import os
import csv
import datetime
from datetime import timedelta
import statistics
from pathlib import Path
import time
import numpy as np
import copy
import math

def GetWeekDictionary(assetDF, include_volume):

    '''
    This piece of code breaks up the daily csv into weeks
    '''

    startBinDatetime, endBinDatetime = datetime.datetime.strptime(_configKeys.STARTPULL, '%d/%m/%Y'), datetime.datetime.strptime(_configKeys.ENDPULL, '%d/%m/%Y')

    countDatetime = startBinDatetime
    #bins = [] might not need this anymore
    datetimeBin = {}

    while (countDatetime < endBinDatetime): # while the count time is not at the last week in the sequence
        datetimeBin[countDatetime] = []
        #bins.append(datetimeBin) we might not need this code anymore
        countDatetime = countDatetime + timedelta(days=7)


    #This first puts the y value into the bins list. This is to give us easy access when trying to move it to the yValues list

    assetWeek = []
    currentBinDate = startBinDatetime

    for ind in assetDF.index:

        # Current date for stock is past current bin.
        if (datetime.datetime.strptime(assetDF['Date'][ind], '%Y-%m-%d') - currentBinDate).days > 7:
            datetimeBin[currentBinDate] = assetWeek
            currentBinDate = currentBinDate + timedelta(days=7)
            if include_volume == True:
                assetWeek = [[datetime.datetime.strptime(assetDF['Date'][ind], '%Y-%m-%d'), assetDF['Open'][ind], assetDF['High'][ind], assetDF['Low'][ind], assetDF['Close'][ind], assetDF['Volume'][ind]]]
            else:
                assetWeek = [[datetime.datetime.strptime(assetDF['Date'][ind], '%Y-%m-%d'), assetDF['Open'][ind], assetDF['High'][ind], assetDF['Low'][ind], assetDF['Close'][ind]]]
        else:
            if include_volume == True:
                assetWeek.append([datetime.datetime.strptime(assetDF['Date'][ind], '%Y-%m-%d'), assetDF['Open'][ind], assetDF['High'][ind], assetDF['Low'][ind], assetDF['Close'][ind], assetDF['Volume'][ind]])
            else:
                assetWeek = [[datetime.datetime.strptime(assetDF['Date'][ind], '%Y-%m-%d'), assetDF['Open'][ind], assetDF['High'][ind], assetDF['Low'][ind], assetDF['Close'][ind]]]

    # We have to do this one more time to get the values from the last week
    datetimeBin[currentBinDate] = assetWeek

    return datetimeBin


def CollapseDictionaryToWeeks(dictionary, name, has_volume):
    elementDict = {'Open':1, 'High':2, 'Low':3, 'Close':4, 'Volume':5}
    asset_df = {"Date" : list(dictionary.keys())}
    if has_volume:
        list_of_elements = ['Open', 'High', 'Low', 'Close', 'Volume']
    else:
        list_of_elements = ['Open', 'High', 'Low', 'Close']

    for element in list_of_elements:
        for statistic in ["average", "max", "min", "volatility", "change"]:
        # for statistic in ["average", "max", "min", "volatility"]:
            elementIndex = elementDict[element]
            week_bin_list = []
            for week in dictionary.keys(): # This assumes the keys are already in chronological order
                elementList = []
                for day in dictionary[week]:
                    elementList.append(day[elementIndex])
                if statistic == "average":
                    week_bin_list.append(statistics.mean(elementList))
                elif statistic == "max":
                    week_bin_list.append(max(elementList))
                elif statistic == "min":
                    week_bin_list.append(min(elementList))
                elif statistic == "volatility": #maybe add another "volatility" statistic??
                    week_bin_list.append(max(elementList) - min(elementList))
                elif statistic == "change":
                        week_bin_list.append(elementList[-1] - elementList[0])
                else:
                    print("something went wrong in CollapseDictionaryToWeeks()")
                    quit()
            asset_df[name + "_" + element + "_" +statistic] = week_bin_list

    week_bin_df = pd.DataFrame(asset_df)

    return week_bin_df


def main():
    '''
    This is the section of code where you can specify what stocks/funds/bonds/etc you'd like to get and how to bin them by week
    Data csv columns:

    Stock: Date, Open, High, Low, Close, Volume, Currency
    Fund: Date, Open, High, Low, Close, Currency
    Bonds: Date, Open, High, Low, Close
    Commodity: Date, Open, High, Low, Close, Volume, Currency

    Note: Thankfully, we likely won't need to use the Currency column (since it is always USD).
    This allows us to use the same CollapseDictionaryToWeeks() function for all asset classes
    '''
    successfulBins = {"Symbol" : [], "Type" : []}
    # We will use the 1successfulPulls.csv to tell us what type of asset is associated with each name/ticker
    reference_df = pd.read_csv("1successfulPulls.csv", low_memory=False)

    for index in reference_df.index:

        name = reference_df["Symbol"][index]
        print(name)
        asset_class = reference_df["Type"][index]

        asset_class_has_volume = False
        if asset_class == "Stock" or asset_class == "Commodity":
            asset_class_has_volume = True

        assetDF = pd.read_csv(os.path.join(Path(_configKeys.DATA_FOLDER), name+".csv"), low_memory=False)

        asset_dictionary = GetWeekDictionary(assetDF, asset_class_has_volume)

        try:
            week_bin_df = CollapseDictionaryToWeeks(asset_dictionary, name, asset_class_has_volume)
            week_bin_df.to_csv(os.path.join(Path(_configKeys.BINNED_FOLDER), name+".csv"), index=False)

            #Update the successful bins dataframe
            successfulBins["Symbol"].append(name)
            successfulBins["Type"].append(asset_class)
        except:
            print(str(name) + " has missing data or doesn't bin correctly")


    df = pd.DataFrame(successfulBins, columns = ["Symbol", "Type"])
    #Creating a sucessful file that includes asset names/tickers
    df.to_csv(_configKeys.SUCCESSFULWEEKBINS, index=False)
    #Now we throw this dataframe into a csv file and add it to a 2successfulWeekBins.csv


if __name__ == "__main__":
    main()
