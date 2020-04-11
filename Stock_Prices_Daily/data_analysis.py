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

from pathlib import Path

def lassoRegressionImplement(stockDF, alpha):
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

    # stockWeek[0] = [[Date, Open, High , Low, Close, Adj...], [...], [...]]
    # stockWeek[1] = [[Date, Open, High , Low, Close, Adj...], [...], [...]]
    # stockWeek[2] = [[Date, Open, High , Low, Close, Adj...], [...], [...]]
    # stockWeek[3] = [[Date, Open, High , Low, Close, Adj...], [...], [...]]
    # stockWeek[4] = [[Date, Open, High , Low, Close, Adj...], [...], [...]]

    ##############################################################################

    '''
    This piece of code is adding the highest price per week into the yValues list.
    This could probably be more efficient if we did a week range pull here using yFinance.
    The only reason why I am not doing that is because pulling using yFinance is slow and we
    would have to extract data from a data frame.
    '''
    # for week in stockWeek:
    #     overallHigh = 0
    #     for day in week:
    #         if day[2] > overallHigh:
    #             overallHigh = day[2]
    #     yValues.append(overallHigh)

    extractWeekly(datetimeBin, high=True, volume=False, volaility = True, volumeAvg = True)

    # [overallHighWeek1, overallHighWeek2, overallHighWeek3...]

    ##############################################################################

    '''
    Because the xValues have to be put into [[],[],[],[]] format. I can't think of a
    good way to put it into that format other than extracting all the different types
    of xValues within the same for loop.

    Stack overflow found me this answer: https://stackoverflow.com/questions/6473679/transpose-list-of-lists
    '''


    for week in stockWeek:
        print(len(week))
    quit()

    # overallVolatityPerWeek() => [overallVolatiltyW1, overallVolatiltyW2, overallVolatiltyW3...]
    #
    # IRXbyWeeks() => [IRXWeek1, IRXWeek2, IRXWeek3...]
    #
    # ...
    #
    # transpose: [overallVolatiltyW1, IRXWeek1, XvalueW1, W1, ...]


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
    df['Feature Name'] = xValueNames
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

############################################################################################

# def extractWeekly(dictionary, highAvg, volumeAvg, lowAvg, highVolaility):
#     extractWeeklyDic = {}
#     if volume == True:
#         extractWeeklyDic[volume] = findVolume(dictionary)
#
#     extractWeeklyDic[]

'''
This is the part that actually runs the code
'''

df = pd.read_csv("successfulPulls.csv", low_memory=False)

allStock = {}

for ind in df.index:
    stockDF = pd.read_csv(os.path.join(Path(configKeys.DATA_FOLDER), df['Symbol'][ind]+'Daily.csv'), low_memory=False)
    allStock[df['Symbol'][ind]] = stockDF

alpha = 0.1
for counter in range(10):
    lassoRegressionImplement(allStock['JNJ'], alpha)
    alpha += 0.1
