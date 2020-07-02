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

def find_asset_class(name, reference_df):
    for index in reference_df.index:
        if reference_df["Symbol"][index] == name:
            return reference_df["Type"][index]
    print("could not find " + name + " in the reference csv")
    quit()

def standardizeSeries(series, window_length):
    #input: a list of values (series), and a time frame (window_length - in weeks)
    #output: standardized list of values
    #maybe we can try this, but usgin a year's worth of "training data" at the beginning of the series?
    standardizedList = []
    for i in range(len(series)):
        if i < window_length - 1:
            continue
        else:
            #lookup 'how to standardize data' and try to understand equation a bit
            #(Data point for week i - (Data point from week (i + 1 - window) to week i)) /  Standard Deviation of data points from week (i + 1 - window) to week i
            standardizedList.append((series[i] - statistics.mean(series[i + 1 - window_length:i + 1])) / statistics.stdev(series[i + 1 - window_length:i + 1]))
    return standardizedList


def StandardizeDF(assetDF, window_length):
    standardizedDict = {}
    colNameList = []
    for colName in assetDF:
        if colName == 'Date' or '%' in colName:
            #no changes to data
            columnList = list(assetDF[colName].values)
            #this will cut off the first weeks to keep it the srame as the other columns
            standardizedDict[colName] = columnList[(window_length-1):]
            colNameList.append(colName)
        else:
            columnList = list(assetDF[colName].values)
            try:
                standardizedSeries = standardizeSeries(columnList, window_length)
                if np.isnan(np.sum(standardizedSeries)) == False and math.isinf(np.sum(standardizedSeries)) == False:
                    standardizedDict[colName] = standardizedSeries
                    colNameList.append(colName)
                else:
                    print(colName)
                    print(standardizedSeries)
            except:
                print("standardizing failed")
                print(colName)
    standardizedDF = pd.DataFrame(standardizedDict, columns = colNameList)
    return standardizedDF

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
    reference_df = pd.read_csv("2successfulWeekBins.csv", low_memory=False)
    window_length = _configKeys.WINDOW_LENGTH
    #index is an asset row
    for index in reference_df.index:
        name = reference_df["Symbol"][index]
        print(name)
        asset_class = reference_df["Type"][index]

        '''
        asset_class_has_volume = False
        if asset_class == "Stock" or asset_class == "Commodity":
            asset_class_has_volume = True
        '''
        assetDF = pd.read_csv(os.path.join(Path(_configKeys.BINNED_FOLDER), name+".csv"), low_memory=False)

        standardizedDF = StandardizeDF(assetDF, window_length)
        standardizedDF.to_csv(os.path.join(Path(_configKeys.STANDARDIZED_FOLDER), name+".csv"), index=False)

        #Update the successful bins dataframe
        successfulBins["Symbol"].append(name)
        successfulBins["Type"].append(asset_class)

    df = pd.DataFrame(successfulBins, columns = ["Symbol", "Type"])
    #Creating a sucessful file that includes asset names/tickers
    df.to_csv(_configKeys.SUCCESSFULSTANDARDIZEDBINS, index=False)
    #Now we throw this dataframe into a csv file and add it to a 2successfulWeekBins.csv


#main()
