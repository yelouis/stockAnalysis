# Import the plotting library
import matplotlib.pyplot as plt

#This file will keep track of all the moving variables and we can slowly add to that file
import _configKeys

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
import datetime
from datetime import timedelta
import statistics
from pathlib import Path
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import time
import numpy as np
import copy
import math


def GetWeekDictionary(stockDF):

    '''
    This piece of code breaks up the stocks into weeks
    '''

    startBinDatetime, endBinDatetime = datetime.datetime.strptime(_configKeys.STARTPULL, '%d/%m/%Y'), datetime.datetime.strptime(_configKeys.ENDPULL, '%d/%m/%Y')

    countDatetime = startBinDatetime
    bins = []
    datetimeBin = {}

    while (countDatetime < endBinDatetime): # while the count time is not at the last week in the sequence
        datetimeBin[countDatetime] = []
        bins.append(datetimeBin)
        countDatetime = countDatetime + timedelta(days=7)


    #This first puts the y value into the bins list. This is to give us easy access when trying to move it to the yValues list

    stockWeek = []
    currentBinDate = startBinDatetime

    for ind in stockDF.index:

        if isinstance(stockDF['Date'][ind], str):
            stockDF.at[ind, 'Date'] = datetime.datetime.strptime(stockDF['Date'][ind], '%Y-%m-%d')
        # Current date for stock is past current bin.
        if (stockDF['Date'][ind] - currentBinDate).days > 7:
            datetimeBin[currentBinDate] = stockWeek
            currentBinDate = currentBinDate + timedelta(days=7)
            stockWeek = [[stockDF['Date'][ind], stockDF['Open'][ind], stockDF['High'][ind], stockDF['Low'][ind], stockDF['Close'][ind], stockDF['Adj Close'][ind], stockDF['Volume'][ind]]]
        else:
            stockWeek.append([stockDF['Date'][ind], stockDF['Open'][ind], stockDF['High'][ind], stockDF['Low'][ind], stockDF['Close'][ind], stockDF['Adj Close'][ind], stockDF['Volume'][ind]])

    # We have to do this one more time to get the values from the last week
    datetimeBin[currentBinDate] = stockWeek

    return datetimeBin


def extractWeekly(dictionary, element, statistic): #extractWeekly(bigdictionary, "open", "average")
    elementDict = {'date':0, 'open':1, 'high':2, 'low':3, 'close':4, 'adj close':5, 'volume':6}
    elementIndex = elementDict[element]
    outputSeries = []

    for week in dictionary.keys(): # This assumes the keys are already in chronological order
        elementList = []
        for day in dictionary[week]:
            elementList.append(day[elementIndex])
        if statistic == "average":
            outputSeries.append(statistics.mean(elementList))
        if statistic == "max":
            outputSeries.append(max(elementList))
        if statistic == "volatility":
            outputSeries.append(max(elementList) - min(elementList))
        if statistic == "change":
            outputSeries.append(elementList[-1] - elementList[0])
    return outputSeries


def main():
    '''
    This is the part that actually runs the code
    '''

    stockDF = pd.read_csv(os.path.join(Path(_configKeys.DATA_FOLDER), "BA.csv"), low_memory=False)

    GetWeekDictionary(stockDF)

main()
