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
from sklearn.metrics import mean_absolute_error

def lassoRegressionImplement(xValues, yValues, xValueNames, yValueName, alpha, beta):
    X_train, X_test, y_train, y_test = train_test_split(xValues, yValues, test_size=0.3, random_state=20)
    x_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=20)

    clf = linear_model.Lasso(alpha=alpha)
    clf.fit(X_train, y_train)
    y_predT = clf.predict(X_train)
    y_pred = clf.predict(X_test)
    y_predV = clf.predict(x_valid)

    madT = mean_absolute_error(y_train, y_predT)
    madV = mean_absolute_error(y_valid, y_predV)
    mad = mean_absolute_error(y_test, y_pred)
    print("MAD:", mad)

    df = pd.DataFrame()
    xValueNames = np.append(xValueNames, ['MAD Train', 'MAD Valid', 'MAD Test'])
    df['Feature Name'] = xValueNames
    column_name = 'Alpha =' + str(alpha) + " Beta =" +str(beta)
    coefficients = clf.coef_
    coefficients = np.append(coefficients, [str(madT), str(madV), str(mad)])
    df[column_name] = coefficients


    # Use xStocks to help specify the contents of the file
    alphaString = format(alpha, '.1f')
    betaString = str(int(beta))
    madString = format(mad, '.2f')

    # These two lines of print statement are just asking whether we did better
    # with our predication than with just a one unit lag.
    y_pred = clf.predict(xValues)
    print(mean_absolute_error(yValues, [yValues[0]] + yValues[:-1]))
    print(mean_absolute_error(yValues, y_pred))

    path = os.path.join(Path(configKeys.LASSO_RESULTS_FOLDER), yValueName + madString + "_mad" + alphaString +"_alpha"+ betaString + "_beta" + '.csv')

    df.to_csv(path)


def readXValues(successfulPullsDic, filter_asset_class):
    #Gets all the standardized xValues out of the csv files.
    allXValues = []
    allXValueNames = []
    for ticker in successfulPullsDic:
        if successfulPullsDic[ticker] != filter_asset_class:
            tickerDF = pd.read_csv(os.path.join(Path(_configKeys.STANDARDIZED_FOLDER), str(ticker)+".csv"), low_memory=False)
            for column_name in tickerDF:
                if column_name != "Date":
                    allXValues.append(list(ticketDF[column_name].values))
                    allXValueNames.append(column_name)

    return (allXValues, allXValueNames)

def main():

    reference_df = pd.read_csv("3successfulStandardizedBins.csv", low_memory=False)
    referenceDict = {}
    for index in reference_df.index:
        name = reference_df["Symbol"][index]
        asset_class = reference_df["Type"][index]
        referenceDict[name] = asset_class

    (xValues, xValueNames) = readXValues(referenceDict, "")

    xValues = list(map(list, zip(*xValues)))

    # Where do I get the yValues? Need to clear this up. Maybe make a configKeys variable for the yValue?
    # Can I just put yValueName in as a configKey variable. I can probably get the yValues as long
    # as I know the yValueName
    yValues = ...something[1:] #Need the [1:] to lag out the first yValue
    yValueName = ...

    alpha = 0.1
    for counter in range(10):
        start_time = time.time()
        lassoRegressionImplement(xValues, yValues, xValueNames, yValueName, alpha, beta)
        print("--- %s seconds ---" % (time.time() - start_time))
        alpha += 0.1

main()
