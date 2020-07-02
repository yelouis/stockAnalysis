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
    #Generalize lassoDF (thinking make a list of DFs for each file containing the ticker within LASSO_RESULTS_FOLDER)

    #this allows us to loop through each one of the different lasso files from 0.3 to 1.2 by incrementing 0.1

    #read in from 4successfulLasso.csv and just iterate through a list of those names to access successfully lassoed filenames
    #floatStepList = list(np.arange(0.3, 1.3, 0.1))
    threshold = _configKeys.THRESHOLD
    successfulLassoDF = pd.read_csv("4successfulLasso.csv", low_memory=False)

    lassoDF = pd.read_csv(os.path.join(Path(_configKeys.LASSO_RESULTS_FOLDER), findBestLassoCSV(_configKeys.YVALUETICKER)), low_memory=False)
    #predictList will contain a list of values
    predictionsDict = makePredictionsDict(lassoDF, threshold)
    standardizedTestingDF = makeStandardizedTestingDF(predictionsDict)
    unstandardizedTestingDF = makeUnstandardizedTestingDF(standardizedTestingDF, window_length)
    unstandardizedTestingDF.to_csv(os.path.join(Path(_configKeys.TESTING_RESULTS_FOLDER), name+"_test_results.csv"))

    '''
    for name in list(successfulLassoDF["FileName"].values):
        #name = str(name)
        lassoDF = pd.read_csv(os.path.join(Path(_configKeys.LASSO_RESULTS_FOLDER), name+".csv"), low_memory=False)
        #predictList will contain a list of values
        predictionsDict = makePredictionsDict(lassoDF, threshold)
        standardizedTestingDF = makeStandardizedTestingDF(predictionsDict)
        unstandardizedTestingDF = makeUnstandardizedTestingDF(standardizedTestingDF, window_length)
        unstandardizedTestingDF.to_csv(os.path.join(Path(_configKeys.TESTING_RESULTS_FOLDER), name+"_test_results.csv"))
    '''
    #lassoDF = pd.read_csv(os.path.join(Path(_configKeys.LASSO_RESULTS_FOLDER), str(_configKeys.YVALUETICKER)+"0.3_alpha13_beta.csv"), low_memory=False)
    '''
    #predictList will contain a list of values
    predictionsDict = makePredictionsDict(lassoDF, threshold)
    standardizedTestingDF = makeStandardizedTestingDF(predictionsDict)
    unstandardizedTestingDF = makeUnstandardizedTestingDF(standardizedTestingDF, window_length)
    '''

    #unstandardizedTestingDF.to_csv(os.path.join(Path(_configKeys.TESTING_RESULTS_FOLDER), str(_configKeys.YVALUETICKER)+"_test_results.csv"))

#Finds the best csv for predictions
def findBestLassoCSV(ticker):
    csvList = os.listdir(_configKeys.LASSO_RESULTS_FOLDER)
    tickerCSVList = []
    csvMADList = []
    for csv in csvList:
        if ticker in csv:
            tickerCSVList.append(csv)
    for csv in tickerCSVList:
        lassoDF = pd.read_csv(os.path.join(Path(_configKeys.LASSO_RESULTS_FOLDER), csv))
        lowAvgMADT = list(lassoDF[_configKeys.YVALUETICKER+"_Low_average_toggles"].values)[0]
        #calculateMeanError(list(testingDF[_configKeys.YVALUETICKER + "_Low_average_Predicted"].values), list(testingDF[_configKeys.YVALUETICKER + "_Low_average_Actual"].values))
        highAvgMADT = list(lassoDF[_configKeys.YVALUETICKER + "_High_average_toggles"].values)[0]
        #calculateMeanError(list(testingDF[_configKeys.YVALUETICKER + "_High_average_Predicted"].values), list(testingDF[_configKeys.YVALUETICKER + "_High_average_Actual"].values))
        avgMADT = (highAvgMADT + lowAvgMADT) / 2
        csvMADList.append((csv, avgMADT))

    csvMADList.sort(key=lambda tup: tup[1])
    print (csvMADList)
    return csvMADList[0][0]



'''
s is standardized predicted and a is unstandardized predicted
We know nothing about a6 or s6

[a1, a2, a3, a4, a5]

for a in a's:
    (a - mean([a's]))/ std.[a's]
[s1, s2, s3, s4, s5]

s6: get from Lasso. Lasso didn't anything that required a6.

how do you get a6?
known: a2, a3, a4, a5

(a6 - mean([a2, a3, a4, a5, a6]))/ std.[a2, a3, a4, a5, a6] = s6
Using a guess a6 (first guess is a5):
calc: (a6 - mean([a2, a3, a4, a5, a6]))/ std.[a2, a3, a4, a5, a6] -> guess s6
    compare with predicted s6 (58)

'''


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
            #TODO: Need to use Cole's list
            # unstandardizedListPredicted = calculate_unstandardized(predList, actualList, window_length)
            for i in range(len(predList)):
                #indexing
                unstandardizedListPredicted.append(Estimate_Unstandardized(predList[i], actualList[i:i+window_length-1], window_length))

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

#makeStandardizedTestingDF makes a dataframe with
# Date = list of Dates
# for each y value name:
# [yValueName_Predicted] = predictionList calculated in the makePredictionsDict function
# [yValueName_Actual] =  actual yValues found in the csv of the stock in question in the STANDARDIZED_FOLDER
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
    dateDF = pd.read_csv(os.path.join(Path(_configKeys.STANDARDIZED_FOLDER), _configKeys.YVALUETICKER+".csv"))
    dates = list(dateDF["Date"].values)
    #print (len(predictionsDict))
    predictionsDict ["Date"] = dates[1:]
    #print (len(predictionsDict))
    xValueNames = list(lassoDF['Feature Name'].values)

    for column in lassoDF.columns:
        if "_coefficients" in str(column):

            predictionList = [] #predictionList is a list where each index corresponds with a week and the value at each index is the
            #summation of the multiplications of every coefficient and the corresponding value of the feature name at the corresponding week

            #count() returns the number of non-NAN rows in a dataframe
            #shape() returns a tuple containing the dimension of the dataframe as: (height, width)

            #Changed initialization of predictionList -------------------------------------
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
            #lookup 'how to standardize data' and try to understand equation a bit
            #(Data point for week i - (Data point from week (i + 1 - window) to week i)) /  Standard Deviation of data points from week (i + 1 - window) to week i
            #std eqn: standardizedList.append((series[i] - statistics.mean(series[i + 1 - window_length:i + 1])) / statistics.stdev(series[i + 1 - window_length:i + 1]))
            unstandardizedList.append(predictionSTDList[i + 1 - window_length] * statistics.stdev(series[i + 1 - window_length: i + 1]) + statistics.mean(series[i + 1 - window_length:i+1]))
    return unstandardizedList





def Calculate_Standardized_Value(series, window_length):
    series = list(map(float, series))
    #lastIndex = len(series) - 1
    if statistics.stdev(series) == 0:
        print("CANNOT DIVIDE BY 0 -- problem standardizing data using eqn")
    standardizedValue = (series[-1] - statistics.mean(series)) / statistics.stdev(series)
    #print(standardizedValue)
    return standardizedValue
    #return standardizeSeries([series], window_length)[-1]


#We could try to get the math better, but it is very challenging when we are using the expected value in the standardization
# ((standardized_value * stdev) + (sum(known_values)/window_length)) / ((window_length-1)/window_length)
# (prediction[i] * stdev)

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

main()
