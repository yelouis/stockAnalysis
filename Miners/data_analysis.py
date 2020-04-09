# Import the plotting library
import matplotlib.pyplot as plt

#This file will keep track of all the moving variables and we can slowly add to that file
import configKeys

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
import numpy as np

from pathlib import Path

def lassoRegressionImplement(df, alpha):
    '''
    stockA, stockB, stockC
    xValues = [[stockA.vol, stockB.vol, stockC.vol],[stockA.volit, stockB.volit, stockC.volit],[stockA.lowPrice, stockB.lowPrice, stockC.lowPrice]]
    yValues = [stockA.highest, stockB.highest, stockC.highest]
    '''

    '''
    Doing this!!!
    stockA
    xValues = [[stockA.volW1, stockA.volW2, stockA.volW3],[stockA.volitW1, stockA.volitW2, stockA.volitW3]]
    yValues = [stockA.highestW2, stockA.highestW3, stockA.highestW4]
    '''
    xValues = []
    yValues = []
    xValueNames = []


    '''
    This bins everything into weeks
    '''

    startBinDatetime, endBinDatetime = datetime.datetime.strptime(configKeys.STARTPULL, '%Y-%m-%d'), datetime.datetime.strptime(configKeys.ENDPULL, '%Y-%m-%d')

    countDatetime = startBinDatetime
    bins = []
    datetimeBin = []

    while (countDatetime < endBinDatetime): # while the count time is not at the last week in the sequence
        datetimeBin = [countDatetime, []] # could add more [] lists if you want to add volume and other criteria
        bins.append(datetimeBin)
        countDatetime = countDatetime + timedelta(days=7)


    #This first puts the y value into the bins list. This is to give us easy access when trying to move it to the yValues list

    head = 0
    stockWeek = []

    stockDF = pd.read_csv(os.path.join(Path(configKeys.DATA_FOLDER), df['Y Value'][0]+'.csv'), low_memory=False)

    for ind in stockDF.index:

        stockDF.at[ind, 'Date'] = datetime.datetime.strptime(stockDF['Date'][ind], '%Y-%m-%d')

        if (stockDF['Date'][ind] - bins[head][0]).days <= 7:
            # We know that we've got the right index
            stockWeek.append(stockDF['Close'][ind])

        else:
            while (stockDF['Date'][ind] - bins[head][0]).days > 7:
                if (len(stockWeek) == 0):
                    bins[head][1].append(None)
                else:
                    bins[head][1].append(statistics.mean(stockWeek))
                head += 1
                stockWeek = []

    # We have to do this one more time to get the values from the last week
    if (len(stockWeek) == 0):
        bins[head][1].append(None)
    else:
        bins[head][1].append(statistics.mean(stockWeek))

    #This for loop then collects all the xValue data into the bins
    for stockNameInd in df['Lagged X Value'].index:

        head = 0
        stockWeek = []

        stockDF = pd.read_csv(os.path.join(Path(configKeys.DATA_FOLDER), df['Lagged X Value'][stockNameInd]+'.csv'), low_memory=False)

        for ind in stockDF.index:

            stockDF.at[ind, 'Date'] = datetime.datetime.strptime(stockDF['Date'][ind], '%Y-%m-%d')

            if (stockDF['Date'][ind] - bins[head][0]).days <= 7:
                # We know that we've got the right index
                stockWeek.append(stockDF['Close'][ind])

            else:
                while (stockDF['Date'][ind] - bins[head][0]).days > 7:
                    if (len(stockWeek) == 0):
                        bins[head][1].append(None)
                    else:
                        bins[head][1].append(statistics.mean(stockWeek))
                    head += 1
                    stockWeek = []

        # We have to do this one more time to get the values from the last week
        if (len(stockWeek) == 0):
            bins[head][1].append(None)
        else:
            bins[head][1].append(statistics.mean(stockWeek))

    print(bins)
    quit()

    '''
    INVERT THE bins LIST!!!!!
    '''

    #xValues = bins[]





    '''
    xValueNames = [yvalue price, yvalue vol, xval1 price, xval1 vol, ...]
    bins = [[datetime, [xvalue1[0], xvalue2[0], xvalue3[0], ...]], [datetime, [...[1]]]]

    xValues = [[stockA.volW1, stockA.volW2, stockA.volW3],[stockA.volitW1, stockA.volitW2, stockA.volitW3]]
    yValues = [stockA.highestW2, stockA.highestW3, stockA.highestW4]
    '''

    ##############################################################################
    '''
    This piece of code breaks up X value stocks into weeks
    '''

    for stockNameInd in df['Lagged X Value'].index:

        stockDays = []
        avgStockWeek = []
        varianceStockWeek = []

        stockDF = pd.read_csv(os.path.join(Path(configKeys.DATA_FOLDER), df['Lagged X Value'][stockNameInd]+'.csv'), low_memory=False)

        for ind in stockDF.index:

            stockDF.at[ind, 'Date'] = datetime.datetime.strptime(stockDF['Date'][ind], '%Y-%m-%d')

            if ind == 0:
                stockDays.append(stockDF['Close'][ind]) #stockDF['Volume'][ind])
            else:
                if (stockDF['Date'][ind] - stockDF['Date'][ind-1]).days >= 2 and len(stockDays) > 1:
                    avgStockWeek.append(statistics.mean(stockDays))
                    varianceStockWeek.append(statistics.variance(stockDays))
                    stockDays = []
                elif ((stockDF['Date'][ind] - stockDF['Date'][ind-1]).days >= 2):
                    stockDays = []
                stockDays.append(stockDF['Close'][ind])
        if len(stockDays) > 1:
            avgStockWeek.append(statistics.mean(stockDays))
            varianceStockWeek.append(statistics.variance(stockDays))

        # This will set the first x values to be the lagged version of the y value (price of the stock in question)
        xValues.append(avgStockWeek[:-1])
        xValues.append(varianceStockWeek[:-1])

        xValueNames.append(df['Lagged X Value'][stockNameInd] + ' average')
        xValueNames.append(df['Lagged X Value'][stockNameInd] + ' variance')





    #This will then break the Y value stock into weeks and add it to yValues and its lag to xValues


    yValues = avgStockWeek[1:]

    print([len(val) for val in xValues])
    quit()

    ##############################################################################


    '''
    Standardize the x values
    '''





    from sklearn.preprocessing import StandardScaler
    scalerX = StandardScaler()
    scalerX.fit(xValues)
    xValues = scalerX.transform(xValues)

    '''
    Regarding standardizing the yValues. We might want to think about standardizing
    it in comparison to the previous week's highs rather than standardizing it to each
    other. That way we would be predicting how much higher or lower the high of next
    week's stock is going to be in relation to the previous week's highs
    '''
    # standardizedYValues = []
    # for target in yValues:
    #     standardizedYValues.append([target])
    # scalerY = StandardScaler()
    # scalerY.fit(standardizedYValues)
    # standardizedYValues = scalerY.transform(standardizedYValues)
    # yValues = []
    # for target in standardizedYValues:
    #     yValues.append(target[0])

    '''
    Split dataset in to testing, validation, and training dataset
    '''

    X_train, X_test, y_train, y_test = train_test_split(xValues, yValues, test_size=0.3, random_state=20)
    x_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=20)

    clf = linear_model.Lasso(alpha=alpha)
    clf.fit(X_train, y_train)
    y_predT = clf.predict(X_train)
    y_pred = clf.predict(X_test)
    y_predV = clf.predict(x_valid)
    from sklearn.metrics import mean_absolute_error
    '''
    Calculate the MAD score for each of the sub datasets
    '''
    madT = mean_absolute_error(y_train, y_predT)
    madV = mean_absolute_error(y_valid, y_predV)
    mad = mean_absolute_error(y_test, y_pred)
    print("MAD:", mad)

    df = pd.DataFrame()
    df['Feature Name'] = name
    column_name = 'Alpha =' + str(alpha)
    df[column_name] = clf.coef_

    '''
    Write the coefficients of each feature into a file
    '''
    path = os.path.join(Dataset.config[configKeys.TABLE_DIRECTORY] + 'lassoResult' +
                        '_length' + str(int(actLength)) + '_wrong' + str(int(wrongAct)) + '_info' +
                        str(int(infoAct)) + '_pause' + str(int(pauseAct)) + '_mcS' + str(int(mcScore)) +
                        '_mcP' + str(int(mcProblem)) + '_pre' + str(int(preScore)) + '_chalAmt' +
                        str(int(challengeAmt)) + '_preScoreM' + str(int(preScoreMedian)) + '_lowPreUser' +
                        str(int(lowPreUsers)) + '_MAD' + str(round(madV, 3)) + '_alpha' + str(round(alpha, 1)) + '.csv')

    df.to_csv(path)

    with open(path, 'a') as fd:
        fd.write("MAD Train: " + str(madT) + '\n')
        fd.write("MAD Valid: " + str(madV) + '\n')
        fd.write('MAD Test: ' + str(mad))

    print(df)

##############################################################################
'''
This is the part that actually runs the code
'''

df = pd.read_csv("XYValues.csv", low_memory=False)

alpha = 0.1
for counter in range(10):
    lassoRegressionImplement(df, alpha)
    alpha += 0.1
