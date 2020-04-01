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

from pathlib import Path

df = pd.read_csv("successfulPulls.csv", low_memory=False)

allStock = {}

for ind in df.index:
    stockDF = pd.read_csv(os.path.join(Path(configKeys.DATA_FOLDER), df['Symbol'][ind]+'Daily.csv'), low_memory=False)
    allStock[df['Symbol'][ind]] = stockDF

alpha = 0.1
for counter in range(10):
    lassoRegressionImplement(allStock, alpha)
    alpha += 0.1

def lassoRegressionImplement(allStock, alpha):
    '''
    stockA, stockB, stockC
    xValues = [[stockA.vol, stockB.vol, stockC.vol],[stockA.volit, stockB.volit, stockC.volit],[stockA.lowPrice, stockB.lowPrice, stockC.lowPrice]]
    yValues = [stockA.highest, stockB.highest, stockC.highest]
    '''

    

    '''
    Standardize the x values
    '''
    from sklearn.preprocessing import StandardScaler
    scalerX = StandardScaler()
    scalerX.fit(xValues)
    xValues = scalerX.transform(xValues)

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
