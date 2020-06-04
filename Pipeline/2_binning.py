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


def CollapseDictionaryToWeeks(dictionary, element, statistic):
    elementDict = {'Date':0, 'Open':1, 'High':2, 'Low':3, 'Close':4, 'Volume':5}
    elementIndex = elementDict[element]
    week_bin_list = []
    date_bin_list = []

    for week in dictionary.keys(): # This assumes the keys are already in chronological order
        elementList = []
        for day in dictionary[week]:
            elementList.append(day[elementIndex])
        if statistic == "average":
            week_bin_list.append(statistics.mean(elementList))
        elif statistic == "max":
            week_bin_list.append(max(elementList))
        elif statistic == "volatility": #maybe add another "volatility" statistic??
            week_bin_list.append(max(elementList) - min(elementList))
        elif statistic == "change":
            week_bin_list.append(elementList[-1] - elementList[0])
        else:
            print("invalid input statistic to CollapseDictionaryToWeeks()")
            quit()
        date_bin_list.append(week)

    week_bin_df = pd.DataFrame({"Date":date_bin_list,
                         "Value":week_bin_list})

    return week_bin_df


def find_asset_class(name, reference_df):
    for index in reference_df.index:
        if reference_df["Symbol"][index] == name:
            return reference_df["Sector"][index]
    print("could not find " + name + " in the reference csv")
    quit()

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
    name_element_stat = [["BA", "Close", "average"],
                        ["Gold", "Close", "max"],
                        ["BA", "Low", "average"],
                        ["BA", "Close", "volatility"],
                        ["The Hartford Midcap Fund Class C", "Close", "average"],
                        ["U.S. 30Y", "Open", "average"]]

    # We will use the 1successfulPulls.csv to tell us what type of asset is associated with each name/ticker
    reference_df = pd.read_csv("1successfulPulls.csv", low_memory=False)

    for asset in name_element_stat:
        name = asset[0]
        element = asset[1]
        stat_to_get = asset[2]

        asset_class = find_asset_class(name, reference_df)
        asset_class_has_volume = False
        if asset_class == "Stock" or asset_class == "Commodity":
            asset_class_has_volume = True

        # check to see if we are trying to get the volume of an asset which does not have volume provided
        if asset_class_has_volume != True and element == "Volume":
            print(name + " does not have an associated volume")
            print("Try choosing a different element")
            quit()

        assetDF = pd.read_csv(os.path.join(Path(_configKeys.DATA_FOLDER), name+".csv"), low_memory=False)

        asset_dictionary = GetWeekDictionary(assetDF, asset_class_has_volume)

        week_bin_df = CollapseDictionaryToWeeks(asset_dictionary, element, stat_to_get)
        print(week_bin_df)

        #Now we throw this dataframe into a csv file and add it to a 2successfulWeekBins.csv


main()
