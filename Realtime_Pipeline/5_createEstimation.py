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

#Finds the best csv for predictions
def findBestLassoCSV():
    successfulLassoDF = pd.read_csv("4successfulLasso.csv", low_memory=False)
    csvList = list(successfulLassoDF["FileName"].values)
    csvMADList = []
    for csv in csvList:
        lassoDF = pd.read_csv(os.path.join(Path(_configKeys.LASSO_RESULTS_FOLDER), csv + ".csv"))
        lowAvgMAD = list(lassoDF[_configKeys.YVALUETICKER+"_Low_average_toggles"].values)[2].strip("mad =")
        #calculateMeanError(list(testingDF[_configKeys.YVALUETICKER + "_Low_average_Predicted"].values), list(testingDF[_configKeys.YVALUETICKER + "_Low_average_Actual"].values))
        highAvgMAD = list(lassoDF[_configKeys.YVALUETICKER + "_High_average_toggles"].values)[2].strip("mad =")
        #calculateMeanError(list(testingDF[_configKeys.YVALUETICKER + "_High_average_Predicted"].values), list(testingDF[_configKeys.YVALUETICKER + "_High_average_Actual"].values))
        avgMAD = (float(highAvgMAD) + float(lowAvgMAD)) / 2
        csvMADList.append((csv, avgMAD))

    csvMADList.sort(key=lambda tup: tup[1])
    return csvMADList[0][0]


'''
makePredictionsDict makes a dictionary holding [Key] = Value:
    [Date] = list of Dates
    for each y value name:
    [yValueName_Predicted] = predictionList calculated in the function
    [yValueName_Actual] = empty list to be overwritten later
                (do this here because it will be easier later when working with data frames to fill in premade spaces)
'''
def makePredictionsDict(lassoDF, threshold):
    predictionsDict = {}
    dateDF = pd.read_csv(os.path.join(Path(_configKeys.STANDARDIZED_FOLDER), _configKeys.YVALUETICKER+".csv"))
    dates = list(dateDF["Date"].values)
    #print (len(predictionsDict))
    predictionsDict["Date"] = dates[1:]
    #print (len(predictionsDict))
    xValueNames = list(lassoDF['Feature Name'].values)

    for column in lassoDF.columns:
        if "_coefficients" in str(column):

            predictionList = []

            predictionList = [0]*(dateDF.shape[0]-1)

            coefficients = list(lassoDF[column].values)
            yValueName = column.split("_")[:-1]
            yValueName = str(yValueName[0]) + "_" + str(yValueName[1]) + "_" + str(yValueName[2])

            for i in range(len(coefficients)):
                coefficient = coefficients[i]
                if abs(coefficient) > threshold:
                    #print (coefficient)
                    #xValueName: stock name/ticker
                    xValueName = lassoDF.iloc[i][1]
                    print(str(_configKeys.STANDARDIZED_FOLDER) + xValueName.split("_")[0]+".csv")
                    featureStockDF = pd.read_csv(os.path.join(Path(_configKeys.STANDARDIZED_FOLDER), xValueName.split("_")[0]+".csv"))

                    #print (str(xValueName))
                    #This code gives us the weekly data for the affecting feature
                    #print (featureStockDF.columns)
                    #print()

                    #Changed accessing correct column ------------------------------

                    #print (str(featureStockCol))
                    featureWeekList = list(featureStockDF[xValueName].values[:-1])
                    #print (len(featureWeekList)) #CORRECT SIZE
                    for x in range(len(featureWeekList[:-1])):
                        #Multiplies this column's values by the coefficient and adds the result to the total prediction list

                        #look into np.arrays
                        #np.array(featureWeekList) * coefficient
                        #predictionList + featureWeekList (if both ar np.arrays)
                        featureWeekDataPoint = featureWeekList[x]
                        predictionList[x] += coefficient * featureWeekDataPoint
                    print(str(xValueName) + " does not exists in current data")

            predictionsDict[yValueName+"_Predicted"] = predictionList
            predictionsDict[yValueName+"_Actual"] = predictionList
    return predictionsDict


def main():
    window_length = _configKeys.WINDOW_LENGTH
    reference_df = pd.read_csv("3successfulStandardizedBins.csv", low_memory=False)
    referenceDict = readSuccessfulFile(reference_df)
    threshold = _configKeys.THRESHOLD

    name = findBestLassoCSV()

    lassoDF = pd.read_csv(os.path.join(Path(_configKeys.LASSO_RESULTS_FOLDER), name + ".csv"), low_memory=False)
    predictionsDict = makePredictionsDict(lassoDF, threshold)
    standardizedTestingDF = makeStandardizedTestingDF(predictionsDict)
    unstandardizedTestingDF = makeUnstandardizedTestingDF(standardizedTestingDF, window_length)
    unstandardizedTestingDF.to_csv(os.path.join(Path(_configKeys.TESTING_RESULTS_FOLDER), name+"_test_results.csv"))

    successfulDict = {"FileName" : [name]}
    df = pd.DataFrame(successfulDict, columns = ["FileName"])
    df.to_csv('5successfulTesting.csv', index=False)
