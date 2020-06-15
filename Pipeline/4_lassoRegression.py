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

def lassoRegressionImplement(xValues, yValues, xValueNames, yValueName, alpha, beta):
    X_train, X_test, y_train, y_test = train_test_split(xValues, yValues, test_size=0.3, random_state=20)
    x_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=20)

    clf = linear_model.Lasso(alpha=alpha, tol=0.01, max_iter=1000000000)
    clf.fit(X_train, y_train)
    y_predT = clf.predict(X_train)
    y_pred = clf.predict(X_test)
    y_predV = clf.predict(x_valid)

    madT = mean_absolute_error(y_train, y_predT)
    madV = mean_absolute_error(y_valid, y_predV)
    mad = mean_absolute_error(y_test, y_pred)
    # print("MAD:", mad)

    df = pd.DataFrame()
    df['Feature Name'] = xValueNames
    coefficients = clf.coef_
    df[str(yValueName)] = coefficients

    df2 = pd.DataFrame()
    df2[str(yValueName)+'_toggles'] = ['madT =' + str(madT), 'madV =' + str(madV), 'mad =' + str(mad), 'Alpha =' + str(alpha), 'Beta =' +str(beta)]

    return pd.concat([df, df2], axis=1, sort=False)


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

def readSuccessfulFile(reference_df):
    referenceDict = {}
    for index in reference_df.index:
        name = reference_df["Symbol"][index]
        asset_class = reference_df["Type"][index]
        referenceDict[name] = asset_class
    return referenceDict

def main():

    reference_df = pd.read_csv("3successfulStandardizedBins.csv", low_memory=False)
    referenceDict = readSuccessfulFile(reference_df)

    (xValues, xValueNames) = readXValues(referenceDict, "")

    xValues = list(map(list, zip(*xValues)))

    beta = _configKeys.WINDOW_LENGTH
    yValueDict = readYValues()

    alpha = .3
    for counter in range(10):
        allYValueResults = pd.DataFrame()
        start_time = time.time()
        for yValueName in yValueDict:
            yValues = yValueDict[yValueName]
            singleYValueResult = lassoRegressionImplement(xValues, yValues, xValueNames, yValueName, alpha, beta)
            allYValueResults = pd.concat([allYValueResults, singleYValueResult], axis=1, sort=False)
        path = os.path.join(Path(_configKeys.LASSO_RESULTS_FOLDER), str(_configKeys.YVALUETICKER) + str(format(alpha, '.1f')) +"_alpha"+ str(int(beta)) + "_beta" + '.csv')
        allYValueResults.to_csv(path)
        print("--- %s seconds ---" % (time.time() - start_time))
        alpha += 0.1
        quit()

    reference_df.to_csv('4successfulLasso.csv', index=False)

main()
