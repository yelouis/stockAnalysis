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

#Finds the best csv for predictions
def findCSV():
    successfulTestingDF = pd.read_csv("5successfulTesting.csv", low_memory=False)
    testingCSV = list(successfulTestingDF["FileName"].values)[0]
    # Having the [0] means that we are only looking at the first value in the 5successfulTesting file
    # This is under the assumption that since we only pick the best alpha from a selected beta,
    #    we will only have one file to calculate limits on
    return testingCSV

def calculateLimitPrice(estimationFileName_df):
    sell_marker_predicted = _configKeys.YVALUETICKER + "_" + _configKeys.SELL_LIMIT_MARKER + "_Predicted"
    buy_marker_predicted = _configKeys.YVALUETICKER + "_" + _configKeys.BUY_LIMIT_MARKER + "_Predicted"
    sell_marker_actual = _configKeys.YVALUETICKER + "_" + _configKeys.SELL_LIMIT_MARKER + "_Actual"
    buy_marker_actual = _configKeys.YVALUETICKER + "_" + _configKeys.BUY_LIMIT_MARKER + "_Actual"

    initialSellPrice = list(estimationFileName_df[sell_marker_predicted].values)[-1]
    initialBuyPrice = list(estimationFileName_df[buy_marker_predicted].values)[-1]

    sellPredictedValues = list(estimationFileName_df[sell_marker_predicted].values)[:-1]
    sellActualValues = list(estimationFileName_df[sell_marker_actual].values)[:-1]

    buyPredictedValues = list(estimationFileName_df[buy_marker_predicted].values)[:-1]
    buyActualValues = list(estimationFileName_df[buy_marker_actual].values)[:-1]

    sellPrice = initialSellPrice - calculateMeanError(sellActualValues, sellPredictedValues)
    buyPrice = initialBuyPrice - calculateMeanError(buyActualValues, buyPredictedValues)

    limitsDict = {"Sell_Price": [sellPrice], "Buy_Price": [buyPrice]}

    return limitsDict

def calculateMeanError(actualList, predictionList):
    totalMean = 0
    for i in range(len(actualList)):
        totalMean += actualList[i] - predictionList[i]
    return totalMean/len(actualList)

def main():
    estimationFileName = findCSV()
    estimationFileName_df = pd.read_csv(os.path.join(Path(_configKeys.TESTING_RESULTS_FOLDER), estimationFileName + "_test_results.csv"), low_memory=False)
    limits_Dict = calculateLimitPrice(estimationFileName_df)
    limits_df = pd.DataFrame(limits_Dict, columns = ["Sell_Price", "Buy_Price"])

    limits_df.to_csv(os.path.join(Path(_configKeys.LIMIT_RESULTS_FOLDER), estimationFileName+"_limit_results.csv"))

if __name__ == "__main__":
    main()
