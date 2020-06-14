#This file will keep track of all the moving variables and we can slowly add to that file
import _configKeys

import pandas as pd
import collections
import copy
import os
import csv
import datetime
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import statistics
from pathlib import Path
import time
import numpy as np
import copy
import math
from sklearn.metrics import mean_absolute_error


def readSuccessfulFile(reference_df):
    referenceDict = {}
    for index in reference_df.index:
        name = reference_df["Symbol"][index]
        asset_class = reference_df["Type"][index]
        referenceDict[name] = asset_class
    return referenceDict

def readXValues(successfulPullsDic, filter_asset_class):
    #Gets all the standardized xValues out of the csv files.
    allXValues = []
    allXValueNames = []
    for ticker in successfulPullsDic:
        if successfulPullsDic[ticker] != filter_asset_class:
            tickerDF = pd.read_csv(os.path.join(Path(_configKeys.STANDARDIZED_FOLDER), str(ticker)+".csv"), low_memory=False)
            for column_name in tickerDF:
                if column_name != "Date" and "%" not in column_name:
                    currentXValue = list(tickerDF[column_name].values)
                    lengthOfCurrentXValue = len(currentXValue)
                    #We need to cut the xValues down by 1 time unit
                    allXValues.append(currentXValue[:lengthOfCurrentXValue-1])
                    allXValueNames.append(column_name)
    return (allXValues, allXValueNames)

def readYValues():
    yValueDF = pd.read_csv(os.path.join(Path(_configKeys.STANDARDIZED_FOLDER), str(_configKeys.YVALUETICKER)+".csv"), low_memory=False)
    allYValues = {}
    for yValueColName in yValueDF:
        if yValueColName != "Date":
            #We need to increase yValues by 1 time unit
            allYValues[yValueColName] = list(yValueDF[yValueColName].values[1:])
    return allYValues


def main():
    reference_df = pd.read_csv("3successfulStandardizedBins.csv", low_memory=False)
    referenceDict = readSuccessfulFile(reference_df)

    (xValues, xValueNames) = readXValues(referenceDict, "")

    xValues = list(map(list, zip(*xValues)))

    #YValueDict is organized such that the key is the statistic that we are interested
    #in and the value is a list containing the yValues.
    yValueDict = readYValues()

    for yValueKey in yValueDict:
        something()


main()
