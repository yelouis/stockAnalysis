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

def makeUnstandardizedTestingDF(stdDF, window_length):

    columnList = list(stdDF.keys())
    ticker = columnList[1].split("_")[0]

    unstdDF = pd.DataFrame(columns = columnList)
    unstdDF["Date"] = stdDF["Date"]
    binnedDF = pd.read_csv(os.path.join(Path(_configKeys.BINNED_FOLDER), ticker+".csv"))

    for i in range(1, len(stdDF.columns)):
        col = stdDF.columns[i]
        colName = col.split("_")[:-1]
        colName = str(colName[0]) + "_" + str(colName[1]) + "_" + str(colName[2])

        #actualList = list(binnedDF[colName].values)

        actualList = list(binnedDF[colName].values)[-(2*_configKeys.WINDOW_LENGTH - 1):]
        actualList.append(0)

        if "Predicted" in col:
            predList = list(stdDF[col].values)

            unstandardizedListPredicted = []
            #TODO: Need to use Cole's list
            # unstandardizedListPredicted = calculate_unstandardized(predList, actualList, window_length)
            for j in range(len(predList)):
                #indexing
                unstandardizedListPredicted.append(Estimate_Unstandardized(predList[j], actualList[j:j+window_length-1], window_length))

            '''
            for i in range(len(predList)):
                unstandardizedValuePredicted = Estimate_Unstandardized(predList[i], actualList[i:i+window_length], window_length)
                unstandardizedListPredicted.append(unstandardizedValuePredicted)
            '''
            #print(str(len(unstandardizedListPredicted)) + "Predicted")
            unstdDF[col] = unstandardizedListPredicted

        if "Actual" in col:
            #print(str(len(actualList)) + "Actual")
            unstdDF[col] = actualList[window_length:]

    return unstdDF

def makeStandardizedTestingDF(predictionsDict):
    columnList = list(predictionsDict.keys())
    print(columnList)
    ticker = columnList[1].split("_")[0]
    actualDF = pd.read_csv(os.path.join(Path(_configKeys.STANDARDIZED_FOLDER), ticker+".csv"))

    for actualColName in columnList:
        if "Actual" in actualColName:
            colName = actualColName.split("_")[:-1]
            colName = str(colName[0]) + "_" + str(colName[1]) + "_" + str(colName[2])
            actualList = list(actualDF[colName].values)
            predictionsDict[actualColName] = actualList[-_configKeys.WINDOW_LENGTH:]
            predictionsDict[actualColName].append(0)
    # for value in predictionsDict.values():
    #    print(len(value))

    stdDF = pd.DataFrame.from_dict(predictionsDict)
    #stdDF.to_csv('JNJstdFileAll.csv', index=False)
    return stdDF

def makePredictionsDict(lassoDF, threshold):
    predictionsDict = {}
    dateDF = pd.read_csv(os.path.join(Path(_configKeys.STANDARDIZED_FOLDER), _configKeys.YVALUETICKER+".csv"))
    dates = list(dateDF["Date"].values)
    #print (len(predictionsDict))

    '''
    PredictionsDict is a dictionary. Within the 'Date' key of the dictionary,
    I am adding an extra week at the end of the Dates to indicate the upcoming week

    Hence the 'predictionsDict["Date"].append(futurePredictionDate)'
    '''

    predictionsDict["Date"] = dates[-_configKeys.WINDOW_LENGTH:]
    futurePredictionDate = datetime.datetime.strftime(datetime.datetime.strptime(predictionsDict["Date"][-1], '%Y-%m-%d') + timedelta(days=7), '%Y-%m-%d')
    predictionsDict["Date"].append(futurePredictionDate)

    #print (len(predictionsDict))
    xValueNames = list(lassoDF['Feature Name'].values)

    for column in lassoDF.columns:
        if "_coefficients" in str(column):

            predictionList = [] #predictionList is a list where each index corresponds with a week and the value at each index is the
            #summation of the multiplications of every coefficient and the corresponding value of the feature name at the corresponding week

            predictionList = [0]* (_configKeys.WINDOW_LENGTH + 1)

            coefficients = list(lassoDF[column].values)

            yValueName = column.split("_")[:-1]
            yValueName = str(yValueName[0]) + "_" + str(yValueName[1]) + "_" + str(yValueName[2])
            #yValueName now becomes something like AAPLStock_Open_max
            #All we did is remove the coefficient part

            for i in range(len(coefficients)):
                coefficient = coefficients[i]
                if abs(coefficient) > threshold:

                    xValueName = lassoDF.iloc[i][1]
                    print(str(_configKeys.STANDARDIZED_FOLDER) + xValueName.split("_")[0]+".csv")
                    featureStockDF = pd.read_csv(os.path.join(Path(_configKeys.STANDARDIZED_FOLDER), xValueName.split("_")[0]+".csv"))

                    featureWeekList = list(featureStockDF[xValueName].values[-(_configKeys.WINDOW_LENGTH+1):])

                    for x in range(len(featureWeekList)):
                        #Multiplies this column's values by the coefficient and adds the result to the total prediction list

                        featureWeekDataPoint = featureWeekList[x]
                        predictionList[x] += coefficient * featureWeekDataPoint



            predictionsDict[yValueName+"_Predicted"] = predictionList
            # Setting yValueName+"_Actual" to predictionList just to make it the same size
            predictionsDict[yValueName+"_Actual"] = predictionList
    return predictionsDict

def Estimate_Unstandardized(standardized_value, known_values, window_length):
    '''
    Nathaniel idea:
        idea for later: maybe lets compare the unstandardized values given from the known values and compare them to these values
    '''

    '''
    known_values: the beta-1 weeks before the week of interest [$12, $14, $13.4, ...] -- ex: if window_length is 13, we want the 12 weeks before and we're going to append an estimate for the
    window_length: the beta value
    standardized_value: the standardized value of the week we are trying to predict

    alg: [$12, $14, $13.4, ...] append x (estimated_value) variable
    keep changing x to get the predicted standardized value as close to the given standardized_value

    x - mean([$12, $14, $13.4, ... x])/ stdev([$12, $14, $13.4, ... x])
    Needs to get as close as possible to standardized_value
    '''

    estimated_value = int(known_values[-1])

    factor = 10 ** (len(str(int(known_values[-1]))) - 1) # Take most recent value as judge of where the prediction could move (+ or - 100% max)

    while(True):

        if factor < 0.001: # done when we know the nearest tenth of a cent
            break

        dif = standardized_value - Calculate_Standardized_Value(known_values + [estimated_value], window_length)

        if dif <= 0:
            for j in range(10):
                if abs(dif) > abs(standardized_value - Calculate_Standardized_Value(known_values + [estimated_value - factor], window_length)):
                    estimated_value -= factor
                    dif = standardized_value - Calculate_Standardized_Value(known_values + [estimated_value], window_length) # update dif
                else:
                    break
        else:
            for j in range(10):
                if abs(dif) > abs(standardized_value - Calculate_Standardized_Value(known_values + [estimated_value + factor], window_length)):
                    estimated_value += factor
                    dif = standardized_value - Calculate_Standardized_Value(known_values + [estimated_value], window_length) # update dif
                else:
                    break

        factor = factor / 10

    return estimated_value

def Calculate_Standardized_Value(series, window_length):
    series = list(map(float, series))
    #lastIndex = len(series) - 1
    if statistics.stdev(series) == 0:
        print("CANNOT DIVIDE BY 0 -- problem standardizing data using eqn")
    standardizedValue = (series[-1] - statistics.mean(series)) / statistics.stdev(series)
    return standardizedValue

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
