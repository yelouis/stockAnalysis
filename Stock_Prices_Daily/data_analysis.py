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
from pathlib import Path
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

def GetWeekDictionary(stockDF):

    '''
    This piece of code breaks up the stocks into weeks
    '''

    startBinDatetime, endBinDatetime = datetime.datetime.strptime(configKeys.STARTPULL, '%Y-%m-%d'), datetime.datetime.strptime(configKeys.ENDPULL, '%Y-%m-%d')

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

def extractWeekly(dictionary, element, statistic):
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
        if statistic == "standard deviation":
            outputSeries.append(statistics.stdev(elementList))
        if statistic == "variance":
            outputSeries.append(statistics.variance(elementList))
        if statistic == "change":
            outputSeries.append(elementList[-1] - elementList[0])
    return outputSeries

    # Intuition: going through the dictionary, look at the specified "statistic" of each week's elements at index (found above in dictionary).
    # Add this to the list to return

def getX(stockDF, element, statistic):
    return extractWeekly(GetWeekDictionary(stockDF), element, statistic)[:-1]

def getY(stockDF, element, statistic):
    return extractWeekly(GetWeekDictionary(stockDF), element, statistic)[1:]

def lassoRegressionImplement(allStock, alpha):
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

    extractWeekly = {}
    extractWeekly[vol] = [stockA.volW1, stockA.volW2, stockA.volW3]
    extractWeekly[volit] = [stockA.volitW1, stockA.volitW2, stockA.volitW3]
    extractWeekly[volAvg] =
    '''
    xValues = []
    yValues = []
    xValueNames = []

    ##############################################################################

    '''
    Maybe try this to make file path and push extractWeekly(GetWeekDictionary()) functions behind the scenes
    '''

    # input stocks/statistics of interest in this list
    xStocks = [["JNJ", "high", "average"],
                ["JNJ", "high", "max"],
                ["JNJ", "close", "average"]]

    for i in xStocks:
        xValues.append(getX(allStock[i[0]], i[1], i[2]))
        xValueNames.append(i[0] + "-" + i[1] + "-" + i[2])

    '''
    # THIS IS THE PART OF CODE WHICH WE MANUALLY CHANGE TO DO ANALYSIS
    xValues = [getX(allStock["JNJ"], "high", "average"), getX(allStock["JNJ"], "high", "average")]
    xValueNames.append("high-average")
    xValueNames.append("high-average")
    '''

    yValues = getY(allStock["JNJ"], "high", "average")




    ##############################################################################
    '''
    Standardizing method. To get any specific index i value in xValues to be standardized
    we find its deviation with respect to all values before it (assuming that all the values
    before it have not already been standardized).

    We create a temp list called tempScalerList so that we are not overwriting the
    xValues with standardized values. tempScalerList is recreated at each itteration.
    '''

    '''
    Instead of standardizing we can just normalize and manually calculate the normalize
    value becaue the calculation is pretty easy.

    https://stackoverflow.com/questions/26785354/normalizing-a-list-of-numbers-in-python

    Standardize vs Normalize:
    https://towardsdatascience.com/normalization-vs-standardization-quantitative-analysis-a91e8a79cebf
    '''

    newXValues = []
    for list in xValues:
        standardizedXList = [0] * len(list)
        print(len(list))
        for i in range(len(list)):
            if i != 0:
                tempScalerList = []
                scalerX = StandardScaler()
                scalerX.fit(xValues[:i])
                tempScalerList = scalerX.transform(xValues[:i])

                # Take a look at tempScalerList after the first itteration. I don't
                # understand why there are so many values.
                # print(tempScalerList)

                if i == 3:
                    print(tempScalerList)
                    quit()

                standardizedXList[i] = tempScalerList[0][i]
        print(standardizedXList)
        quit()
        newXValues.append(standardizedXList)

    xValues = newXValues
    quit()

    #maybe we can try this, but usgin a year's worth of "training data" at the beginning of the series?
    newXValues = []
    for list in xValues:
        standardizedXList = [0] * len(list)
        print(len(list))
        for i in range(len(list)):
            standardizedXList.append((list[i] - statistics.mean(list[:i + 1])) / statistics.stdev(list[:i + 1]))
        print(standardizedXList)
        quit()
        newXValues.append(standardizedXList)
    xValues = newXValues
    quit()


    '''
    Because the xValues have to be put into [[],[],[],[]] format. I can't think of a
    good way to put it into that format other than extracting all the different types
    of xValues within the same for loop.

    Stack overflow found me this answer: https://stackoverflow.com/questions/6473679/transpose-list-of-lists
    '''

    xValues = list(map(list, zip(*xValues)))



    '''
    Standardize the x values

    Do we need to standardize? If so, how? We don't want information about future weeks
    because calculating a standard deviation will take into account future weeks.

    Reason why we should standardize: MAD Value easier to understand when we are looking
    by deviation.
    '''
    # from sklearn.preprocessing import StandardScaler
    # scalerX = StandardScaler()
    # scalerX.fit(xValues)
    # xValues = scalerX.transform(xValues)

    '''
    Regarding standardizing the yValues. We might want to think about standardizing
    it in comparison to the previous week's highs rather than standardizing it to each
    other. That way we would be predicting how much higher or lower the high of next
    week's stock is going to be in relation to the previous week's highs
    '''

    # for i in range(len(yValues)):
    #     quit()

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
    df['Feature Name'] = xValueNames
    column_name = 'Alpha =' + str(alpha)
    df[column_name] = clf.coef_

    '''
    Write the coefficients of each feature into a file
    '''

    # Use xStocks to help specify the contents of the file
    path = os.path.join(Path(configKeys.OUTPUT_FOLDER), "name" + '.csv')

    df.to_csv(path)

    with open(path, 'a') as fd:
        fd.write("MAD Train: " + str(madT) + '\n')
        fd.write("MAD Valid: " + str(madV) + '\n')
        fd.write('MAD Test: ' + str(mad))

    print(df)

############################################################################################


def main():
    '''
    This is the part that actually runs the code
    '''

    df = pd.read_csv("successfulPulls.csv", low_memory=False)

    allStock = {}

    for ind in df.index:
        stockDF = pd.read_csv(os.path.join(Path(configKeys.DATA_FOLDER), df['Symbol'][ind]+'Daily.csv'), low_memory=False)
        allStock[df['Symbol'][ind]] = stockDF

    alpha = 0.1
    for counter in range(1):
        lassoRegressionImplement(allStock, alpha)
        alpha += 0.1

main()
