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

def main():

    window_length = _configKeys.WINDOW_LENGTH
    reference_df = pd.read_csv("3successfulStandardizedBins.csv", low_memory=False)
    referenceDict = readSuccessfulFile(reference_df)


    #lassoDF: reading in the df file from the lasso folder
    #lassoDF = pd.read_csv("Gold0.3_alpha13_beta.csv", low_memory=False)
    lassoDF = pd.read_csv(os.path.join(Path(_configKeys.LASSO_RESULTS_FOLDER), str(_configKeys.YVALUETICKER)+"0.3_alpha13_beta.csv"), low_memory=False)
    threshold = _configKeys.THRESHOLD

    #predictList will contain a list of values
    predictionsDict = makePredictionsDict(lassoDF, threshold)
    standardizedTestingDF = makeStandardizedTestingDF(predictionsDict)
    unstandardizedTestingDF = makeUnstandardizedTestingDF(standardizedTestingDF, window_length)
    unstandardizedTestingDF.to_csv(os.path.join(Path(_configKeys.TESTING_RESULTS_FOLDER), str(_configKeys.YVALUETICKER)+"_test_results.csv"))




'''
    (xValues, xValueNames) = readXValues(referenceDict, "")

    xValues = list(map(list, zip(*xValues)))
    #YValueDict is organized such that the key is the statistic that we are interested
    #in and the value is a list containing the yValues.
    yValueDict = readYValues()
    for yValueKey in yValueDict:
        something()
'''

def makeUnstandardizedTestingDF(stdDF, window_length):
    columnList = list(stdDF.keys())
    ticker = columnList[1].split("_")[0]

    unstdDF = pd.DataFrame(columns = columnList)
    unstdDF["Date"] = stdDF["Date"]


    binnedDF = pd.read_csv(os.path.join(Path(_configKeys.BINNED_FOLDER), ticker+".csv"))
    for i in range(1, len(stdDF.columns)):
        col = stdDF.columns[i]
        #print (col):
        colName = col.split("_")[:-1]
        #print (colName)
        colName = str(colName[0]) + "_" + str(colName[1]) + "_" + str(colName[2])
        actualList = list(binnedDF[colName].values)

        if "Predicted" in col:
            actualCol = stdDF.columns[i+1]
            predList = list(stdDF[col].values)
            #actualList = list(stdDF[actualCol].values)

            unstandardizedListPredicted = []
            unstandardizedListPredicted = calculate_unstandardized(predList, actualList, window_length)
            '''
            for i in range(len(predList)):
                unstandardizedValuePredicted = Estimate_Unstandardized(predList[i], actualList[i:i+window_length], window_length)
                unstandardizedListPredicted.append(unstandardizedValuePredicted)
            '''
            print(str(len(unstandardizedListPredicted)) + "Predicted")
            unstdDF[col] = unstandardizedListPredicted

        if "Actual" in col:
            print(str(len(actualList)) + "Actual")
            unstdDF[col] = actualList[:-window_length]

    return unstdDF

#makeStandardizedTestingDF makes a dataframe with
# Date = list of Dates
# for each y value name:
# [yValueName_Predicted] = predictionList calculated in the makePredictionsDict function
# [yValueName_Actual] =  actual yValues found in the csv of the stock in question in the STANDARDIZED_FOLDER
def makeStandardizedTestingDF(predictionsDict):
    columnList = list(predictionsDict.keys())
    #print(len(columnList))
    ticker = columnList[1].split("_")[0]
    actualDF = pd.read_csv(os.path.join(Path(_configKeys.STANDARDIZED_FOLDER), ticker+".csv"))

    for actualColName in columnList:
        if "Actual" in actualColName:
            colName = actualColName.split("_")[:-1]
            colName = str(colName[0]) + "_" + str(colName[1]) + "_" + str(colName[2])
            actualList = list(actualDF[colName].values)
            predictionsDict[actualColName] = actualList[1:]
    #for value in predictionsDict.values():
    #    print(len(value))

    stdDF = pd.DataFrame.from_dict(predictionsDict)
    return stdDF

#makePredictionsDict makes a dictionary holding [Key] = Value:
    # [Date] = list of Dates
    # for each y value name:
    # [yValueName_Predicted] = predictionList calculated in the function
    # [yValueName_Actual] = empty list to be overwritten later (do this here because it will be easier later when working with data frames to fill in premade spaces)
def makePredictionsDict(lassoDF, threshold):
    predictionsDict = {}
    dateDF = pd.read_csv(os.path.join(Path(_configKeys.STANDARDIZED_FOLDER), _configKeys.YVALUETICKER.capitalize()+".csv"))
    dates = list(dateDF["Date"].values)
    #print (len(predictionsDict))
    predictionsDict ["Date"] = dates[1:]
    #print (len(predictionsDict))
    xValueNames = list(lassoDF['Feature Name'].values)

    for column in lassoDF.columns:
        #tell louis to change
        if "_coefficents" in str(column):

            predictionList = [] #predictionList is a list where each index corresponds with a week and the value at each index is the
            #summation of the multiplications of every coefficient and the corresponding value of the feature name at the corresponding week

            #count() returns the number of non-NAN rows in a dataframe
            #shape() returns a tuple containing the dimension of the dataframe as: (height, width)

            for i in range(dateDF.shape[0]-1):
                predictionList.append(0)

            coefficients = list(lassoDF[column].values)
            yValueName = column.split("_")[:-1]
            yValueName = str(yValueName[0]) + "_" + str(yValueName[1]) + "_" + str(yValueName[2])
            for i in range(len(coefficients)):
                coefficient = coefficients[i]
                if abs(coefficient) > threshold:
                    #print (coefficient)
                    #xValueName: stock name/ticker
                    xValueName = lassoDF.iloc[i][1]
                    featureStockDF = pd.read_csv(os.path.join(Path(_configKeys.STANDARDIZED_FOLDER), xValueName.split("_")[0]+".csv"))

                    #print (str(xValueName))
                    #This code gives us the weekly data for the affecting feature
                    #print (featureStockDF.columns)
                    #print()
                    for featureStockCol in featureStockDF.columns:
                        if featureStockCol == xValueName:
                            #print (str(featureStockCol))
                            featureWeekList = list(featureStockDF[featureStockCol].values[:-1])
                            #print (len(featureWeekList)) #CORRECT SIZE
                            for x in range(len(featureWeekList[:-1])):
                                #Multiplies this column's values by the coefficient and adds the result to the total prediction list
                                featureWeekDataPoint = featureWeekList[x]
                                predictionList[x] += coefficient * featureWeekDataPoint


            predictionsDict[yValueName+"_Predicted"] = predictionList
            predictionsDict[yValueName+"_Actual"] = predictionList
    return predictionsDict



'''
pseudocode:

initialize a total prediction list with all 0's
for every Feature Name in the Feature Name column:
    1) check the coefficient to see if its absolut value is over a threshhold (0.1?)
    2) if above the threshhold: (ex: ALIM_Volume_average | 1.3)
        Open the ALIM.csv file from the standardized folder
        get the element_statistic column (Volume_average) and index from [0:length - 1]
        multiply this column's values by the coefficient (1.3) and add them to the total prediction list
---------------------------------------------------------------------
add the total prediction list to a data frame (noting the value it is predicting and the fact that it is the prediction)
add the actual standardized values of the value we are trying to predict

Note: yValueName consists of yStock_Element_Statistic

df1:
{Date: [List of Dates (Yavlues so it wil probably start with 4/3/16, aka index 1)]
 'yValueName'_Predicted_Standardized: [total prediction list]
 'yValueName'_Actual_Standardized: [(Go to yStock file and find the matching yStock_Element_Statistic)[1:end] (in folder 3)]
}

We want to get df2 from df1:
{Date: [List of Dates (Yavlues so it wil probably start with 4/3/16, aka index 1)]
 'yValueName'_Predicted: [total prediction list (Unstandardized)]
 'yValueName'_Actual: [(Go to yStock file and find the matching yStock_Element_Statistic)[window_length(beta) + 1:end] (in folder 2)]
}

To do this, we need to use the unstandardizing function.

Final step:
Initialize a df3 that is empty

concatnate all df2s into df3

Write df3 into an csv with the name: 'yStock'_alpha_beta.csv @ location 5testing_Results
Example: Gold0.3_alpha13_beta.csv @ location 5testing_Results
Really it is the just the same name as the file you read in to get the coefficient

'''

def calculate_unstandardized(predictionSTDList, series, window_length):
    '''
    print(str(len(predictionSTDList)) + "predicted")
    print(str(len(series)) + "actual")
    print(str(window_length) + "WINDOW_LENGTH")
    '''
    unstandardizedList = []
    for i in range(len(series) - 1):
        if i < window_length - 1:
            continue
        else:
            if i + 1 - window_length >= 196:
                print("oooooooo")
            #lookup 'how to standardize data' and try to understand equation a bit
            #(Data point for week i - (Data point from week (i + 1 - window) to week i)) /  Standard Deviation of data points from week (i + 1 - window) to week i
            #std eqn: standardizedList.append((series[i] - statistics.mean(series[i + 1 - window_length:i + 1])) / statistics.stdev(series[i + 1 - window_length:i + 1]))
            unstandardizedList.append(predictionSTDList[i + 1 - window_length] * statistics.stdev(series[i + 1 - window_length: i + 1]) + statistics.mean(series[i + 1 - window_length:i+1]))
    return unstandardizedList


def Calculate_Standardized_Value(series_to_standardize, window_length):
    return standardizeSeries([series_to_standardize], window_length)[0][0]

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

#We could try to get the math better, but it is very challenging when we are using the expected value in the standardization
# ((standardized_value * stdev) + (sum(known_values)/window_length)) / ((window_length-1)/window_length)
# (prediction[i] * stdev)

def Estimate_Unstandardized(standardized_value, known_values, window_length):

    '''
    known_values: the beta-1 weeks before the week of interest [$12, $14, $13.4, ...]
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

main()
