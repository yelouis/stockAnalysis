import createOrderPreservedActionVectors
import sys
import json
import calculateNumClusters
import sklearn
from sklearn import random_projection
import dataAnalysis
import pathlib
from Dataset import iNeuron,ChemV, iNeuronReclassified
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import configKeys
import subprocess
import calculatePatternMetrics
from sklearn import linear_model
import statistics
import csv
import math
import time
import numpy


def lassoRegressionChallenge(Dataset, alpha, stopChallenge, startChallenge, actLength=False, wrongAct=False, infoAct=False,
                    pauseAct=False, mcScore=False, mcProblem=False, challengeAmt = False, preScore=False, lowPreUsers=False,
                    justChallenge=False, preScoreMedian = False):
    completingUser, newChallengeList = Dataset.completingChallengeUser(stopChallenge, startChallenge)
    mcDictionary = Dataset.multipleChoiceTable()
    challengeAmountDictionary = Dataset.challengeAmountDictionary()
    xValues = []
    yValues = []
    stdevDictionary = {}
    meanDictionary = {}
    name = []

    if lowPreUsers:
        lowUserList = Dataset.belowAveragePreStudent(stopChallenge, startChallenge)
        completingUser = list(set(completingUser) & set(lowUserList))


    '''
    Starts another function if justChallenge is set to true. justChallenge looks at the coefficient of a user doing a specific
    challenge
    '''

    if justChallenge:
        df, mad, madV, madT = lassoRegressionOnJustChallenges(Dataset, stopChallenge, startChallenge, completingUser, alpha)
        path = os.path.join(Dataset.config[configKeys.TABLE_DIRECTORY] + 'lassoResult' +
                            '_length' + str(int(actLength)) + '_wrong' + str(int(wrongAct)) + '_info' +
                            str(int(infoAct)) + '_pauseAct' + str(int(pauseAct)) + '_mcS' + str(int(mcScore)) +
                            '_mcP' + str(int(mcProblem)) + '_preS' + str(int(preScore)) + '_chalAmt' +
                            str(int(challengeAmt)) + '_preScoreM' + str(int(preScoreMedian)) + '_lowPreUser' +
                            str(int(lowPreUsers)) + '_justChal' + str(int(justChallenge)) + '_MAD' + str(round(madV, 3))
                            + '_alpha' + str(round(alpha, 1)) + '.csv')
        df.to_csv(path)
        with open(path, 'a') as fd:
            fd.write("MAD Train: " + str(madT) + '\n')
            fd.write("MAD Valid: " + str(madV) + '\n')
            fd.write('MAD Test: ' + str(mad))
        return

    '''
    First loop is to create the stdevDictionary and mean dictionary to store the stdev and mean such that it can later be
    used to bring values that are 3 standard deviations out back towards the mean.
    '''
    for challenge in newChallengeList:
        lengthSTDevList = []
        wrongSTDevList = []
        infoSTDevList = []
        pauseSTDevList = []
        mcScoreSTDevList = []
        mcProblemSTDevList = []
        actionVector = Dataset.getActionVectors(challenge)
        actionIndex = Dataset.getActionIndex(challenge)

        reversedActionIndex = {}
        for action in actionIndex:
            reversedActionIndex[actionIndex[action]] = action

        for user in completingUser:
            wrongAction = 0
            infoAction = 0
            pauseAction = 0
            if not math.isnan(Dataset.getGainScore(user)):
                for action in actionVector[user]:
                    if 'Wrong' in reversedActionIndex[action]:
                        wrongAction += 1
                    if 'Wanted more information/Explored properties' in reversedActionIndex[action]:
                        infoAction += 1
                    if 'PAUSE' in reversedActionIndex[action]:
                        pauseAction += 1
                lengthSTDevList.append(len(actionVector[user]))
                wrongSTDevList.append(wrongAction)
                infoSTDevList.append(infoAction)
                mcScoreSTDevList.append(mcDictionary[challenge][user][1])
                mcProblemSTDevList.append(mcDictionary[challenge][user][0])
                pauseSTDevList.append(pauseAction)

        lengthSTDev = statistics.stdev(lengthSTDevList)
        lengthMean = statistics.mean(lengthSTDevList)
        wrongSTDev = statistics.stdev(wrongSTDevList)
        wrongMean = statistics.mean(wrongSTDevList)
        infoSTDev = statistics.stdev(infoSTDevList)
        infoMean = statistics.mean(infoSTDevList)
        mcScoreSTDev = statistics.stdev(mcScoreSTDevList)
        mcScoreMean = statistics.mean(mcScoreSTDevList)
        mcProblemSTDev = statistics.stdev(mcProblemSTDevList)
        mcProblemMean = statistics.mean(mcProblemSTDevList)
        pauseSTDev = statistics.stdev(pauseSTDevList)
        pauseMean = statistics.mean(pauseSTDevList)

        stdevDictionary[challenge] = []
        stdevDictionary[challenge].append(3*lengthSTDev)
        stdevDictionary[challenge].append(3*wrongSTDev)
        stdevDictionary[challenge].append(3*infoSTDev)
        stdevDictionary[challenge].append(3*mcScoreSTDev)
        stdevDictionary[challenge].append(3*mcProblemSTDev)
        stdevDictionary[challenge].append(3*pauseSTDev)

        meanDictionary[challenge] = []
        meanDictionary[challenge].append(lengthMean)
        meanDictionary[challenge].append(wrongMean)
        meanDictionary[challenge].append(infoMean)
        meanDictionary[challenge].append(mcScoreMean)
        meanDictionary[challenge].append(mcProblemMean)
        meanDictionary[challenge].append(pauseMean)

        '''
        Adds name to list (called name) that is used to create the dataframe for the coefficients
        '''
        if actLength:
            name.append(challenge + " actionLength")
        if wrongAct:
            name.append(challenge + " wrongAction")
        if infoAct:
            name.append(challenge + " infoAction")
        if mcScore:
            name.append(challenge + " mcScore")
        if mcProblem:
            name.append(challenge + " mcProblem")
        if pauseAct:
            name.append(challenge + " pauseAction")

    '''
    Gets the mean and standard deviations for features that are not challenge specific
    '''
    preScoreSTDevList = []
    challengeAmtSTDevList = []
    preScoreMedianSTDevList = []
    for user in completingUser:
        if not math.isnan(Dataset.getGainScore(user)):
            preScoreSTDevList.append(Dataset.getPretestScore(user))
            challengeAmtSTDevList.append(challengeAmountDictionary[user])

    for user in completingUser:
        if not math.isnan(Dataset.getGainScore(user)):
            preScoreMedianSTDevList.append(abs(Dataset.getPretestScore(user) - statistics.median(preScoreSTDevList)))

    preScoreSTDev = statistics.stdev(preScoreSTDevList)
    preScoreMean = statistics.mean(preScoreSTDevList)
    stdevDictionary['preScore'] = 3*preScoreSTDev
    meanDictionary['preScore'] = preScoreMean

    challengeAmtSTDev = statistics.stdev(challengeAmtSTDevList)
    challengeAmtMean = statistics.mean(challengeAmtSTDevList)
    stdevDictionary['challengeAmt'] = 3*challengeAmtSTDev
    meanDictionary['challengeAmt'] = challengeAmtMean

    preScoreMedianSTDev = statistics.stdev(preScoreMedianSTDevList)
    preScoreMedianMean = statistics.mean(preScoreMedianSTDevList)
    stdevDictionary['preScoreMedian'] = 3*preScoreMedianSTDev
    meanDictionary['preScoreMedian'] = preScoreMedianMean

    if preScore:
        name.append("preScore")

    if preScoreMedian:
        name.append("preScoreMedian")

    if challengeAmt:
        name.append("challengeAmt")

    '''
    This loop is to save the x and y values in correct format for lasso regression. y values are the gains and the x values
    are [actionVector length for challenge 1, wrong actions in actionVector for challenge 1, more info action in actionVector
    for challenge 1, actionVector length for challenge 2, ... etc] for each user that has completed all the challenges in our
    challenge of interest.
    '''

    sortUser = {}
    for user in completingUser:
        if not math.isnan(Dataset.getGainScore(user)):
            userXValue = []
            for challenge in newChallengeList:
                actionVector = Dataset.getActionVectors(challenge)
                actionIndex = Dataset.getActionIndex(challenge)
                wrongAction = 0
                infoAction = 0
                pauseAction = 0

                reversedActionIndex = {}
                for action in actionIndex:
                    reversedActionIndex[actionIndex[action]] = action

                for action in actionVector[user]:
                    if 'Wrong' in reversedActionIndex[action]:
                        wrongAction += 1
                    if 'Wanted more information/Explored properties' in reversedActionIndex[action]:
                        infoAction += 1
                    if 'PAUSE' in reversedActionIndex[action]:
                        pauseAction += 1

                if actLength:
                    if abs(len(actionVector[user]) - meanDictionary[challenge][0]) > stdevDictionary[challenge][0]:
                        if meanDictionary[challenge][0] > len(actionVector[user]):
                            userXValue.append(meanDictionary[challenge][0] - stdevDictionary[challenge][0])
                        else:
                            userXValue.append(meanDictionary[challenge][0] + stdevDictionary[challenge][0])
                    else:
                        userXValue.append(len(actionVector[user]))

                if wrongAct:
                    if abs(wrongAction - meanDictionary[challenge][1]) > stdevDictionary[challenge][1]:
                        if meanDictionary[challenge][1] > wrongAction:
                            userXValue.append(meanDictionary[challenge][1] - stdevDictionary[challenge][1])
                        else:
                            userXValue.append(meanDictionary[challenge][1] + stdevDictionary[challenge][1])
                    else:
                        userXValue.append(wrongAction)

                if infoAct:
                    if abs(infoAction - meanDictionary[challenge][2]) > stdevDictionary[challenge][2]:
                        if meanDictionary[challenge][2] > infoAction:
                            userXValue.append(meanDictionary[challenge][2] - stdevDictionary[challenge][2])
                        else:
                            userXValue.append(meanDictionary[challenge][2] + stdevDictionary[challenge][2])
                    else:
                        userXValue.append(infoAction)

                if mcScore:
                    if abs(mcDictionary[challenge][user][1] - meanDictionary[challenge][3]) > stdevDictionary[challenge][3]:
                        if meanDictionary[challenge][3] > mcDictionary[challenge][user][1]:
                            userXValue.append(meanDictionary[challenge][3] - stdevDictionary[challenge][3])
                        else:
                            userXValue.append(meanDictionary[challenge][3] + stdevDictionary[challenge][3])
                    else:
                        userXValue.append(mcDictionary[challenge][user][1])

                if mcProblem:
                    if abs(mcDictionary[challenge][user][0] - meanDictionary[challenge][4]) > stdevDictionary[challenge][4]:
                        if meanDictionary[challenge][4] > mcDictionary[challenge][user][0]:
                            userXValue.append(meanDictionary[challenge][4] - stdevDictionary[challenge][4])
                        else:
                            userXValue.append(meanDictionary[challenge][4] + stdevDictionary[challenge][4])
                    else:
                        userXValue.append(mcDictionary[challenge][user][0])

                if pauseAct:
                    if abs(pauseAction - meanDictionary[challenge][5]) > stdevDictionary[challenge][5]:
                        if meanDictionary[challenge][5] > pauseAction:
                            userXValue.append(meanDictionary[challenge][5] - stdevDictionary[challenge][5])
                        else:
                            userXValue.append(meanDictionary[challenge][5] + stdevDictionary[challenge][5])
                    else:
                        userXValue.append(pauseAction)

            if preScore:
                if abs(Dataset.getPretestScore(user) - meanDictionary['preScore']) > stdevDictionary['preScore']:
                    if meanDictionary['preScore'] > Dataset.getPretestScore(user):
                        userXValue.append(meanDictionary['preScore'] - stdevDictionary['preScore'])
                    else:
                        userXValue.append(meanDictionary['preScore'] + stdevDictionary['preScore'])
                else:
                    userXValue.append(Dataset.getPretestScore(user))

            if preScoreMedian:
                if abs(abs(Dataset.getPretestScore(user) - statistics.median(preScoreSTDevList)) - meanDictionary['preScoreMedian']) > stdevDictionary['preScoreMedian']:
                    if meanDictionary['preScoreMedian'] > abs(Dataset.getPretestScore(user) - statistics.median(preScoreSTDevList)):
                        userXValue.append(meanDictionary['preScoreMedian'] - stdevDictionary['preScoreMedian'])
                    else:
                        userXValue.append(meanDictionary['preScoreMedian'] + stdevDictionary['preScoreMedian'])
                else:
                    userXValue.append(abs(Dataset.getPretestScore(user) - statistics.median(preScoreSTDevList)))

            if challengeAmt:
                if abs(challengeAmountDictionary[user] - meanDictionary['challengeAmt']) > stdevDictionary['challengeAmt']:
                    if meanDictionary['challengeAmt'] > challengeAmountDictionary[user]:
                        userXValue.append(meanDictionary['challengeAmt'] - stdevDictionary['challengeAmt'])
                    else:
                        userXValue.append(meanDictionary['challengeAmt'] + stdevDictionary['challengeAmt'])
                else:
                    userXValue.append(challengeAmountDictionary[user])

            sortUser[user] = [Dataset.getGainScore(user), userXValue]

    for key, value in sorted(sortUser.items()):
        yValues.append(value[0])
        xValues.append(value[1])

    from sklearn.preprocessing import StandardScaler
    scalerX = StandardScaler()
    scalerX.fit(xValues)
    xValues = scalerX.transform(xValues)


    '''
    The commented out code is for when making the target value into standard deviations rather than absolute y values.
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

    X_train, X_test, y_train, y_test = train_test_split(xValues, yValues, test_size=0.3, random_state=20)
    x_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=20)

    '''
    Lasso Regression done here
    '''
    clf = linear_model.Lasso(alpha=alpha)
    clf.fit(X_train, y_train)
    y_predT = clf.predict(X_train)
    y_pred = clf.predict(X_test)
    y_predV = clf.predict(x_valid)
    from sklearn.metrics import mean_absolute_error
    madT = mean_absolute_error(y_train, y_predT)
    madV = mean_absolute_error(y_valid, y_predV)
    mad = mean_absolute_error(y_test, y_pred)
    print("MAD:", mad)

    '''
    Coefficient dataframe created here
    '''
    df = pd.DataFrame()
    df['Feature Name'] = name
    column_name = 'Alpha =' + str(alpha)
    df[column_name] = clf.coef_

    path = os.path.join(Dataset.config[configKeys.TABLE_DIRECTORY] + 'lassoResult' +
                        '_length' + str(int(actLength)) + '_wrong' + str(int(wrongAct)) + '_info' +
                        str(int(infoAct)) + '_pauseAct' + str(int(pauseAct)) + '_mcS' + str(int(mcScore)) +
                        '_mcP' + str(int(mcProblem)) + '_preS' + str(int(preScore)) + '_chalAmt' +
                        str(int(challengeAmt)) + '_preScoreM' + str(int(preScoreMedian)) + '_lowPreUser' +
                        str(int(lowPreUsers)) + '_justChal' + str(int(justChallenge)) + '_MAD' + str(round(madV, 3))
                        + '_alpha' + str(round(alpha, 1)) + '.csv')

    df.to_csv(path)

    with open(path, 'a') as fd:
        fd.write("MAD Train: " + str(madT) + '\n')
        fd.write("MAD Valid: " + str(madV) + '\n')
        fd.write('MAD Test: ' + str(mad))

    print(df)

def lassoRegressionBin(Dataset, alpha, stopChallenge1, startChallenge1, stopChallenge2, startChallenge2, actLength=False, wrongAct=False, infoAct=False,
                    pauseAct=False, mcScore=False, mcProblem=False, challengeAmt = False, preScore=False, lowPreUsers=False,
                    preScoreMedian = False):
    '''

    :param Dataset: The dataset
    :param alpha: alpha value. Should be in lassoRegressionRunner which tests many alpha values and after all the test
    you can pick the best alpha value to run on
    :param stopChallenge1: stop challenge for bin 1
    :param startChallenge1: start challenge for bin 1
    :param stopChallenge2: stop challenge for bin 2
    :param startChallenge2: start challenge for bin 2
    :param actLength: boolean to see if you want to run this feature or not
    :param wrongAct: boolean to see if you want to run this feature or not
    :param infoAct: boolean to see if you want to run this feature or not
    :param pauseAct: boolean to see if you want to run this feature or not
    :param mcScore: boolean to see if you want to run this feature or not
    :param mcProblem: boolean to see if you want to run this feature or not
    :param challengeAmt: boolean to see if you want to run this feature or not
    :param preScore: boolean to see if you want to run this feature or not
    :param lowPreUsers: boolean to see if you want to limit your sample size to just students who have low pre scores
    :param preScoreMedian: boolean to see if you want to run this feature or not
    :return:
    '''

    completingUser, newChallengeList1, newChallengeList2 = Dataset.allChallengeBin(stopChallenge1, startChallenge1, stopChallenge2, startChallenge2)
    newChallengeList = [newChallengeList1, newChallengeList2]
    mcDictionary = Dataset.multipleChoiceTable()
    challengeAmountDictionary = Dataset.challengeAmountDictionary()
    xValues = []
    yValues = []
    stdevDictionary = {}
    meanDictionary = {}
    name = []

    if lowPreUsers:
        lowUserList = Dataset.belowAveragePreStudent(stopChallenge2, startChallenge1)
        completingUser = list(set(completingUser) & set(lowUserList))

    '''
    First loop is to create the stdevDictionary and store the mean and standard deviation for the features of each
    challenge Bin. stdevDictionary is save such that stdevDictionary[challenge][3*lengthSTDev, 3*wrongSTDev,
    3*infoSTDev]. The means are also saved in a dictionary. This checks the amount of standard deviations out later on in the code.
    '''
    userFeature = {}
    loops = 0
    for challengeBin in newChallengeList:
        if loops == 0:
            challengeBinName = 'bin1'
        elif loops == 1:
            challengeBinName = 'bin2'
        loops += 1
        lengthSTDevList = []
        wrongSTDevList = []
        infoSTDevList = []
        pauseSTDevList = []
        mcScoreSTDevList = []
        mcProblemSTDevList = []
        for user in completingUser:
            userFeature[user] = {}
            userFeature[user][challengeBinName] = {}
            totalLength = 0
            wrongAction = 0
            infoAction = 0
            pauseAction = 0
            totalMCScore = 0
            totalMCProblem = 0
            for challenge in challengeBin:
                actionVector = Dataset.getActionVectors(challenge)
                actionIndex = Dataset.getActionIndex(challenge)

                if user in actionVector:
                    reversedActionIndex = {}
                    for action in actionIndex:
                        reversedActionIndex[actionIndex[action]] = action

                    if not math.isnan(Dataset.getGainScore(user)):
                        for action in actionVector[user]:
                            if 'Wrong' in reversedActionIndex[action]:
                                wrongAction += 1
                            if 'Wanted more information/Explored properties' in reversedActionIndex[action]:
                                infoAction += 1
                            if 'PAUSE' in reversedActionIndex[action]:
                                pauseAction += 1

                        totalLength += len(actionVector[user])
                        totalMCScore += mcDictionary[challenge][user][1]
                        totalMCProblem += mcDictionary[challenge][user][0]
            lengthSTDevList.append(totalLength)
            wrongSTDevList.append(wrongAction)
            infoSTDevList.append(infoAction)
            mcScoreSTDevList.append(totalMCScore)
            mcProblemSTDevList.append(totalMCProblem)
            pauseSTDevList.append(pauseAction)

            userFeature[user][challengeBinName]['totalLength'] = totalLength
            userFeature[user][challengeBinName]['wrongAction'] = wrongAction
            userFeature[user][challengeBinName]['infoAction'] = infoAction
            userFeature[user][challengeBinName]['totalMCScore'] = totalMCScore
            userFeature[user][challengeBinName]['totalMCProblem'] = totalMCProblem
            userFeature[user][challengeBinName]['pauseAction'] = pauseAction

        lengthSTDev = statistics.stdev(lengthSTDevList)
        lengthMean = statistics.mean(lengthSTDevList)
        wrongSTDev = statistics.stdev(wrongSTDevList)
        wrongMean = statistics.mean(wrongSTDevList)
        infoSTDev = statistics.stdev(infoSTDevList)
        infoMean = statistics.mean(infoSTDevList)
        mcScoreSTDev = statistics.stdev(mcScoreSTDevList)
        mcScoreMean = statistics.mean(mcScoreSTDevList)
        mcProblemSTDev = statistics.stdev(mcProblemSTDevList)
        mcProblemMean = statistics.mean(mcProblemSTDevList)
        pauseSTDev = statistics.stdev(pauseSTDevList)
        pauseMean = statistics.mean(pauseSTDevList)

        stdevDictionary[challengeBinName] = []
        stdevDictionary[challengeBinName].append(3*lengthSTDev)
        stdevDictionary[challengeBinName].append(3*wrongSTDev)
        stdevDictionary[challengeBinName].append(3*infoSTDev)
        stdevDictionary[challengeBinName].append(3*mcScoreSTDev)
        stdevDictionary[challengeBinName].append(3*mcProblemSTDev)
        stdevDictionary[challengeBinName].append(3*pauseSTDev)

        meanDictionary[challengeBinName] = []
        meanDictionary[challengeBinName].append(lengthMean)
        meanDictionary[challengeBinName].append(wrongMean)
        meanDictionary[challengeBinName].append(infoMean)
        meanDictionary[challengeBinName].append(mcScoreMean)
        meanDictionary[challengeBinName].append(mcProblemMean)
        meanDictionary[challengeBinName].append(pauseMean)

        if actLength:
            name.append(challengeBinName + " actionLength")
        if wrongAct:
            name.append(challengeBinName + " wrongAction")
        if infoAct:
            name.append(challengeBinName + " infoAction")
        if mcScore:
            name.append(challengeBinName + " mcScore")
        if mcProblem:
            name.append(challengeBinName + " mcProblem")
        if pauseAct:
            name.append(challengeBinName + " pauseAction")

    preScoreSTDevList = []
    challengeAmtSTDevList = []
    preScoreMedianSTDevList = []
    for user in completingUser:
        if not math.isnan(Dataset.getGainScore(user)):
            preScoreSTDevList.append(Dataset.getPretestScore(user))
            challengeAmtSTDevList.append(challengeAmountDictionary[user])

    for user in completingUser:
        if not math.isnan(Dataset.getGainScore(user)):
            preScoreMedianSTDevList.append(abs(Dataset.getPretestScore(user) - statistics.median(preScoreSTDevList)))

    preScoreSTDev = statistics.stdev(preScoreSTDevList)
    preScoreMean = statistics.mean(preScoreSTDevList)
    stdevDictionary['preScore'] = 3*preScoreSTDev
    meanDictionary['preScore'] = preScoreMean

    challengeAmtSTDev = statistics.stdev(challengeAmtSTDevList)
    challengeAmtMean = statistics.mean(challengeAmtSTDevList)
    stdevDictionary['challengeAmt'] = 3*challengeAmtSTDev
    meanDictionary['challengeAmt'] = challengeAmtMean

    preScoreMedianSTDev = statistics.stdev(preScoreMedianSTDevList)
    preScoreMedianMean = statistics.mean(preScoreMedianSTDevList)
    stdevDictionary['preScoreMedian'] = 3*preScoreMedianSTDev
    meanDictionary['preScoreMedian'] = preScoreMedianMean

    if preScore:
        name.append("preScore")

    if preScoreMedian:
        name.append("preScoreMedian")

    if challengeAmt:
        name.append("challengeAmt")

    '''
    The individual action values are saved in userFeature[user][challengeBin][action] during the first pass around.
    We are now saving it to a new dictionary where we won't have to care about challengeBin. We multiply the two
    action values from the different challengeBin together to make the userMultiDic.
    '''
    userMultiDic = {}
    for user in userFeature:
        userMultiDic[user] = {}
        userMultiDic[user]['actionLength'] = 1
        userMultiDic[user]['wrongAction'] = 1
        userMultiDic[user]['infoAction'] = 1
        userMultiDic[user]['mcScore'] = 1
        userMultiDic[user]['mcProblem'] = 1
        userMultiDic[user]['pauseAction'] = 1
        for challengeBin in userFeature[user]:
            userMultiDic[user]['actionLength'] *= userFeature[user][challengeBin]['totalLength']
            userMultiDic[user]['wrongAction'] *= userFeature[user][challengeBin]['wrongAction']
            userMultiDic[user]['infoAction'] *= userFeature[user][challengeBin]['infoAction']
            userMultiDic[user]['mcScore'] *= userFeature[user][challengeBin]['totalMCScore']
            userMultiDic[user]['mcProblem'] *= userFeature[user][challengeBin]['totalMCProblem']
            userMultiDic[user]['pauseAction'] *= userFeature[user][challengeBin]['pauseAction']

    '''
    Are appending all values from the userMultiDic to a list so that we can calculate the standard deviation and mean
    of them.
    '''

    actionLengthMultiSTDevList = []
    wrongActionMultiSTDevList = []
    infoActionMultiSTDevList = []
    mcScoreMultiSTDevList = []
    mcProblemMultiSTDevList = []
    pauseActionMultiSTDevList = []
    for user in completingUser:
        if not math.isnan(Dataset.getGainScore(user)):
            actionLengthMultiSTDevList.append(userMultiDic[user]['actionLength'])
            wrongActionMultiSTDevList.append(userMultiDic[user]['wrongAction'])
            infoActionMultiSTDevList.append(userMultiDic[user]['infoAction'])
            mcScoreMultiSTDevList.append(userMultiDic[user]['mcScore'])
            mcProblemMultiSTDevList.append(userMultiDic[user]['mcProblem'])
            pauseActionMultiSTDevList.append(userMultiDic[user]['pauseAction'])

    actionLengthMultiSTDev = statistics.stdev(actionLengthMultiSTDevList)
    actionLengthMultiMean = statistics.mean(actionLengthMultiSTDevList)
    stdevDictionary['actionLengthMulti'] = 3*actionLengthMultiSTDev
    meanDictionary['actionLengthMulti'] = actionLengthMultiMean

    wrongActionMultiSTDev = statistics.stdev(wrongActionMultiSTDevList)
    wrongActionMultiMean = statistics.mean(wrongActionMultiSTDevList)
    stdevDictionary['wrongMulti'] = 3 * wrongActionMultiSTDev
    meanDictionary['wrongMulti'] = wrongActionMultiMean

    infoActionMultiSTDev = statistics.stdev(infoActionMultiSTDevList)
    infoActionMultiMean = statistics.mean(infoActionMultiSTDevList)
    stdevDictionary['infoMulti'] = 3 * infoActionMultiSTDev
    meanDictionary['infoMulti'] = infoActionMultiMean

    mcScoreMultiSTDev = statistics.stdev(mcScoreMultiSTDevList)
    mcScoreMultiMean = statistics.mean(mcScoreMultiSTDevList)
    stdevDictionary['mcScoreMulti'] = 3*mcScoreMultiSTDev
    meanDictionary['mcScoreMulti'] = mcScoreMultiMean

    mcProblemMultiSTDev = statistics.stdev(mcProblemMultiSTDevList)
    mcProblemMultiMean = statistics.mean(mcProblemMultiSTDevList)
    stdevDictionary['mcProblemMulti'] = 3 * mcProblemMultiSTDev
    meanDictionary['mcProblemMulti'] = mcProblemMultiMean

    pauseActionMultiSTDev = statistics.stdev(pauseActionMultiSTDevList)
    pauseActionMultiMean = statistics.mean(pauseActionMultiSTDevList)
    stdevDictionary['pauseMulti'] = 3 * pauseActionMultiSTDev
    meanDictionary['pauseMulti'] = pauseActionMultiMean

    if actLength:
        name.append("actionLengthMulti")

    if wrongAct:
        name.append("wrongActionMulti")

    if infoAct:
        name.append("infoActionMulti")

    if mcScore:
        name.append("mcScoreMulti")

    if mcProblem:
        name.append("mcProblemMulti")

    if pauseAct:
        name.append("pauseActionMulti")

    '''
    Calculates (for user) whether whether or not their features in each bin is within 3 standard deviations of the mean.
    If so, append it as an x value and append the y value as the gain score.
    '''
    sortUser = {}
    for user in completingUser:
        if not math.isnan(Dataset.getGainScore(user)):
            userXValue = []
            loops = 0
            for challengeBin in newChallengeList:
                if loops == 0:
                    challengeBinName = 'bin1'
                elif loops == 1:
                    challengeBinName = 'bin2'
                loops += 1
                totalLength = 0
                wrongAction = 0
                infoAction = 0
                pauseAction = 0
                totalMCScore = 0
                totalMCProblem = 0
                for challenge in challengeBin:
                    actionVector = Dataset.getActionVectors(challenge)
                    actionIndex = Dataset.getActionIndex(challenge)

                    if user in actionVector:
                        reversedActionIndex = {}
                        for action in actionIndex:
                            reversedActionIndex[actionIndex[action]] = action

                        for action in actionVector[user]:
                            if 'Wrong' in reversedActionIndex[action]:
                                wrongAction += 1
                            if 'Wanted more information/Explored properties' in reversedActionIndex[action]:
                                infoAction += 1
                            if 'PAUSE' in reversedActionIndex[action]:
                                pauseAction += 1

                        totalLength += len(actionVector[user])
                        totalMCScore += mcDictionary[challenge][user][1]
                        totalMCProblem += mcDictionary[challenge][user][0]

                if actLength:
                    if abs(totalLength - meanDictionary[challengeBinName][0]) > stdevDictionary[challengeBinName][0]:
                        if meanDictionary[challengeBinName][0] > totalLength:
                            userXValue.append(meanDictionary[challengeBinName][0] - stdevDictionary[challengeBinName][0])
                        else:
                            userXValue.append(meanDictionary[challengeBinName][0] + stdevDictionary[challengeBinName][0])
                    else:
                        userXValue.append(totalLength)

                if wrongAct:
                    if abs(wrongAction - meanDictionary[challengeBinName][1]) > stdevDictionary[challengeBinName][1]:
                        if meanDictionary[challengeBinName][1] > wrongAction:
                            userXValue.append(meanDictionary[challengeBinName][1] - stdevDictionary[challengeBinName][1])
                        else:
                            userXValue.append(meanDictionary[challengeBinName][1] + stdevDictionary[challengeBinName][1])
                    else:
                        userXValue.append(wrongAction)

                if infoAct:
                    if abs(infoAction - meanDictionary[challengeBinName][2]) > stdevDictionary[challengeBinName][2]:
                        if meanDictionary[challengeBinName][2] > infoAction:
                            userXValue.append(meanDictionary[challengeBinName][2] - stdevDictionary[challengeBinName][2])
                        else:
                            userXValue.append(meanDictionary[challengeBinName][2] + stdevDictionary[challengeBinName][2])
                    else:
                        userXValue.append(infoAction)

                if mcScore:
                    if abs(totalMCScore - meanDictionary[challengeBinName][3]) > \
                            stdevDictionary[challengeBinName][3]:
                        if meanDictionary[challengeBinName][3] > totalMCScore:
                            userXValue.append(meanDictionary[challengeBinName][3] - stdevDictionary[challengeBinName][3])
                        else:
                            userXValue.append(meanDictionary[challengeBinName][3] + stdevDictionary[challengeBinName][3])
                    else:
                        userXValue.append(totalMCScore)

                if mcProblem:
                    if abs(totalMCProblem - meanDictionary[challengeBinName][4]) > \
                            stdevDictionary[challengeBinName][4]:
                        if meanDictionary[challengeBinName][4] > totalMCProblem:
                            userXValue.append(meanDictionary[challengeBinName][4] - stdevDictionary[challengeBinName][4])
                        else:
                            userXValue.append(meanDictionary[challengeBinName][4] + stdevDictionary[challengeBinName][4])
                    else:
                        userXValue.append(totalMCProblem)

                if pauseAct:
                    if abs(pauseAction - meanDictionary[challengeBinName][5]) > stdevDictionary[challengeBinName][5]:
                        if meanDictionary[challengeBinName][5] > pauseAction:
                            userXValue.append(meanDictionary[challengeBinName][5] - stdevDictionary[challengeBinName][5])
                        else:
                            userXValue.append(meanDictionary[challengeBinName][5] + stdevDictionary[challengeBinName][5])
                    else:
                        userXValue.append(pauseAction)

            if preScore:
                if abs(Dataset.getPretestScore(user) - meanDictionary['preScore']) > stdevDictionary['preScore']:
                    if meanDictionary['preScore'] > Dataset.getPretestScore(user):
                        userXValue.append(meanDictionary['preScore'] - stdevDictionary['preScore'])
                    else:
                        userXValue.append(meanDictionary['preScore'] + stdevDictionary['preScore'])
                else:
                    userXValue.append(Dataset.getPretestScore(user))

            if preScoreMedian:
                if abs(abs(Dataset.getPretestScore(user) - statistics.median(preScoreSTDevList)) -
                       meanDictionary['preScoreMedian']) > stdevDictionary['preScoreMedian']:
                    if meanDictionary['preScoreMedian'] > abs(
                            Dataset.getPretestScore(user) - statistics.median(preScoreSTDevList)):
                        userXValue.append(meanDictionary['preScoreMedian'] - stdevDictionary['preScoreMedian'])
                    else:
                        userXValue.append(meanDictionary['preScoreMedian'] + stdevDictionary['preScoreMedian'])
                else:
                    userXValue.append(abs(Dataset.getPretestScore(user) - statistics.median(preScoreSTDevList)))

            if challengeAmt:
                if abs(challengeAmountDictionary[user] - meanDictionary['challengeAmt']) > \
                        stdevDictionary['challengeAmt']:
                    if meanDictionary['challengeAmt'] > challengeAmountDictionary[user]:
                        userXValue.append(meanDictionary['challengeAmt'] - stdevDictionary['challengeAmt'])
                    else:
                        userXValue.append(meanDictionary['challengeAmt'] + stdevDictionary['challengeAmt'])
                else:
                    userXValue.append(challengeAmountDictionary[user])

            if actLength:
                if abs(userMultiDic[user]['actionLength'] - meanDictionary['actionLengthMulti']) > stdevDictionary['actionLengthMulti']:
                    if meanDictionary['actionLengthMulti'] > userMultiDic[user]['actionLength']:
                        userXValue.append(meanDictionary['actionLengthMulti'] - stdevDictionary['actionLengthMulti'])
                    else:
                        userXValue.append(meanDictionary['actionLengthMulti'] + stdevDictionary['actionLengthMulti'])
                else:
                    userXValue.append(userMultiDic[user]['actionLength'])

            if wrongAct:
                if abs(userMultiDic[user]['wrongAction'] - meanDictionary['wrongMulti']) > stdevDictionary['wrongMulti']:
                    if meanDictionary['wrongMulti'] > userMultiDic[user]['wrongAction']:
                        userXValue.append(meanDictionary['wrongMulti'] - stdevDictionary['wrongMulti'])
                    else:
                        userXValue.append(meanDictionary['wrongMulti'] + stdevDictionary['wrongMulti'])
                else:
                    userXValue.append(userMultiDic[user]['wrongAction'])

            if infoAct:
                if abs(userMultiDic[user]['infoAction'] - meanDictionary['infoMulti']) > stdevDictionary['infoMulti']:
                    if meanDictionary['infoMulti'] > userMultiDic[user]['infoAction']:
                        userXValue.append(meanDictionary['infoMulti'] - stdevDictionary['infoMulti'])
                    else:
                        userXValue.append(meanDictionary['infoMulti'] + stdevDictionary['infoMulti'])
                else:
                    userXValue.append(userMultiDic[user]['infoAction'])

            if mcScore:
                if abs(userMultiDic[user]['mcScore'] - meanDictionary['mcScoreMulti']) > stdevDictionary['mcScoreMulti']:
                    if meanDictionary['mcScoreMulti'] > userMultiDic[user]['mcScore']:
                        userXValue.append(meanDictionary['mcScoreMulti'] - stdevDictionary['mcScoreMulti'])
                    else:
                        userXValue.append(meanDictionary['mcScoreMulti'] + stdevDictionary['mcScoreMulti'])
                else:
                    userXValue.append(userMultiDic[user]['mcScore'])

            if mcProblem:
                if abs(userMultiDic[user]['mcProblem'] - meanDictionary['mcProblemMulti']) > stdevDictionary['mcProblemMulti']:
                    if meanDictionary['mcProblemMulti'] > userMultiDic[user]['mcProblem']:
                        userXValue.append(meanDictionary['mcProblemMulti'] - stdevDictionary['mcProblemMulti'])
                    else:
                        userXValue.append(meanDictionary['mcProblemMulti'] + stdevDictionary['mcProblemMulti'])
                else:
                    userXValue.append(userMultiDic[user]['mcProblem'])

            if pauseAct:
                if abs(userMultiDic[user]['pauseAction'] - meanDictionary['pauseMulti']) > stdevDictionary['pauseMulti']:
                    if meanDictionary['pauseMulti'] > userMultiDic[user]['pauseAction']:
                        userXValue.append(meanDictionary['pauseMulti'] - stdevDictionary['pauseMulti'])
                    else:
                        userXValue.append(meanDictionary['pauseMulti'] + stdevDictionary['pauseMulti'])
                else:
                    userXValue.append(userMultiDic[user]['pauseAction'])

            sortUser[user] = [Dataset.getGainScore(user), userXValue]

    for key, value in sorted(sortUser.items()):
        yValues.append(value[0])
        xValues.append(value[1])

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

def lassoRegressionOnJustChallenges(Dataset, stopChallenge, startChallenge, userList, alpha):
    '''
    Returns the coefficient of the how much doing a specific challenge has on your gain score.
    JustChallenges cannot be run with any other features because it uses a seperate set of students. The other features
    require that the student be on all the challenges in order to ensure constant feature length but justChallenges
    uses all students as long as they did one challenge.
    :param Dataset: Dataset
    :param stopChallenge: stopChallenge
    :param startChallenge: startChallenge
    :param userList: list of user to look at
    :param alpha: alpha value to run lasso on
    :return:
    '''

    completingUsers, newChallengeList = Dataset.allChallengeUsers(stopChallenge, startChallenge)
    completingUsers = userList
    xValues = []
    yValues = []
    name = []
    for challenge in newChallengeList:
        name.append(challenge)

    sortUser = {}
    for user in completingUsers:
        if not math.isnan(Dataset.getGainScore(user)):
            userXValue = []
            for challenge in newChallengeList:
                actionVector = Dataset.getActionVectors(challenge)
                if user in actionVector:
                    userXValue.append(1)
                else:
                    userXValue.append(0)

            sortUser[user] = [Dataset.getGainScore(user), userXValue]

    for key, value in sorted(sortUser.items()):
        yValues.append(value[0])
        xValues.append(value[1])

    from sklearn.preprocessing import StandardScaler
    scalerX = StandardScaler()
    scalerX.fit(xValues)
    xValues = scalerX.transform(xValues)

    X_train, X_test, y_train, y_test = train_test_split(xValues, yValues, test_size=0.2, random_state=20)
    x_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=20)


    clf = linear_model.Lasso(alpha=alpha)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_predV = clf.predict(x_valid)
    y_predT = clf.predict(X_train)


    from sklearn.metrics import mean_absolute_error
    madV = mean_absolute_error(y_valid, y_predV)
    mad = mean_absolute_error(y_test, y_pred)
    madT = mean_absolute_error(y_train, y_predT)
    print("MAD:", mad)

    df = pd.DataFrame()
    df['Feature Name'] = name
    column_name = 'Alpha = ' + str(alpha)
    df[column_name] = clf.coef_

    return df, mad, madV, madT

def lassoRegressionRunner(Dataset):
    '''
    Loop that runs lassoRegression on alpha values from 0.1 to 1
    :param Dataset: Dataset
    :return:
    '''
    start_time = time.time()
    alpha = 0.1
    for counter in range(10):
        # lassoRegressionChallenge(Dataset, alpha, "listenUpChallenge", "connectTheNeuronsChallenge",
        #                 actLength=False, wrongAct=False, infoAct=False, pauseAct=False, mcScore=False, mcProblem=False,
        #                 preScore=False, challengeAmt=False, lowPreUsers=False, justChallenge=False, preScoreMedian=True)
        # lassoRegressionChallenge(Dataset, alpha, "listenUpChallenge", "connectTheNeuronsChallenge",
        #                 actLength=False, wrongAct=False, infoAct=False, pauseAct=False, mcScore=False, mcProblem=False,
        #                 preScore=False, challengeAmt=False, lowPreUsers=False, justChallenge=True, preScoreMedian=False)
        # lassoRegressionChallenge(Dataset, alpha, "listenUpChallenge", "connectTheNeuronsChallenge",
        #                 actLength=False, wrongAct=False, infoAct=False, pauseAct=False, mcScore=False, mcProblem=False,
        #                 preScore=False, challengeAmt=True, lowPreUsers=False, justChallenge=False, preScoreMedian=False)
        # lassoRegressionChallenge(Dataset, alpha, "listenUpChallenge", "connectTheNeuronsChallenge",
        #                 actLength=False, wrongAct=False, infoAct=False, pauseAct=False, mcScore=False, mcProblem=False,
        #                 preScore=True, challengeAmt=False, lowPreUsers=False, justChallenge=False, preScoreMedian=False)
        # lassoRegressionChallenge(Dataset, alpha, "listenUpChallenge", "connectTheNeuronsChallenge",
        #                 actLength=False, wrongAct=False, infoAct=False, pauseAct=False, mcScore=False, mcProblem=True,
        #                 preScore=False, challengeAmt=False, lowPreUsers=False, justChallenge=False, preScoreMedian=False)
        # lassoRegressionChallenge(Dataset, alpha, "listenUpChallenge", "connectTheNeuronsChallenge",
        #                 actLength=False, wrongAct=False, infoAct=False, pauseAct=False, mcScore=True, mcProblem=False,
        #                 preScore=False, challengeAmt=False, lowPreUsers=False, justChallenge=False, preScoreMedian=False)
        # lassoRegressionChallenge(Dataset, alpha, "listenUpChallenge", "connectTheNeuronsChallenge",
        #                 actLength=False, wrongAct=False, infoAct=False, pauseAct=True, mcScore=False, mcProblem=False,
        #                 preScore=False, challengeAmt=False, lowPreUsers=False, justChallenge=False, preScoreMedian=False)
        # lassoRegressionChallenge(Dataset, alpha, "listenUpChallenge", "connectTheNeuronsChallenge",
        #                 actLength=False, wrongAct=False, infoAct=True, pauseAct=False, mcScore=False, mcProblem=False,
        #                 preScore=False, challengeAmt=False, lowPreUsers=False, justChallenge=False, preScoreMedian=False)
        # lassoRegressionChallenge(Dataset, alpha, "listenUpChallenge", "connectTheNeuronsChallenge",
        #                 actLength=False, wrongAct=True, infoAct=False, pauseAct=False, mcScore=False, mcProblem=False,
        #                 preScore=False, challengeAmt=False, lowPreUsers=False, justChallenge=False, preScoreMedian=False)
        # lassoRegressionChallenge(Dataset, alpha, "listenUpChallenge", "connectTheNeuronsChallenge",
        #                 actLength=True, wrongAct=False, infoAct=False, pauseAct=False, mcScore=False, mcProblem=False,
        #                 preScore=False, challengeAmt=False, lowPreUsers=False, justChallenge=False, preScoreMedian=False)

        lassoRegressionChallenge(Dataset, alpha, "listenUpChallenge", "connectTheNeuronsChallenge",
                        actLength=True, wrongAct=True, infoAct=True, pauseAct=True, mcScore=True, mcProblem=True,
                        preScore=False, challengeAmt=True, lowPreUsers=False, justChallenge=False, preScoreMedian=False)
        alpha += 0.1

    # alpha = 0.1
    # for counter in range(10):
        # lassoRegressionBin(Dataset, alpha, "flexBicepsChallenge", "connectTheNeuronsChallenge",
        #                    "listenUpChallenge", "feelTheSensationChallenge",
        #                    actLength=True, wrongAct=False, infoAct=False,
        #                    pauseAct=False, mcScore=False, mcProblem=False, challengeAmt=False, preScore=False,
        #                    lowPreUsers=False,
        #                    preScoreMedian=False)
        # lassoRegressionBin(Dataset, alpha, "flexBicepsChallenge", "connectTheNeuronsChallenge",
        #                    "listenUpChallenge", "feelTheSensationChallenge",
        #                    actLength=False, wrongAct=True, infoAct=False,
        #                    pauseAct=False, mcScore=False, mcProblem=False, challengeAmt=False, preScore=False,
        #                    lowPreUsers=False,
        #                    preScoreMedian=False)
        # lassoRegressionBin(Dataset, alpha, "flexBicepsChallenge", "connectTheNeuronsChallenge",
        #                    "listenUpChallenge", "feelTheSensationChallenge",
        #                    actLength=False, wrongAct=False, infoAct=True,
        #                    pauseAct=False, mcScore=False, mcProblem=False, challengeAmt=False, preScore=False,
        #                    lowPreUsers=False,
        #                    preScoreMedian=False)
        # lassoRegressionBin(Dataset, alpha, "flexBicepsChallenge", "connectTheNeuronsChallenge",
        #                    "listenUpChallenge", "feelTheSensationChallenge",
        #                    actLength=False, wrongAct=False, infoAct=False,
        #                    pauseAct=True, mcScore=False, mcProblem=False, challengeAmt=False, preScore=False,
        #                    lowPreUsers=False,
        #                    preScoreMedian=False)
        # lassoRegressionBin(Dataset, alpha, "flexBicepsChallenge", "connectTheNeuronsChallenge",
        #                    "listenUpChallenge", "feelTheSensationChallenge",
        #                    actLength=False, wrongAct=False, infoAct=False,
        #                    pauseAct=False, mcScore=True, mcProblem=False, challengeAmt=False, preScore=False,
        #                    lowPreUsers=False,
        #                    preScoreMedian=False)
        # lassoRegressionBin(Dataset, alpha, "flexBicepsChallenge", "connectTheNeuronsChallenge",
        #                    "listenUpChallenge", "feelTheSensationChallenge",
        #                    actLength=False, wrongAct=False, infoAct=False,
        #                    pauseAct=False, mcScore=False, mcProblem=True, challengeAmt=False, preScore=False,
        #                    lowPreUsers=False,
        #                    preScoreMedian=False)
        # lassoRegressionBin(Dataset, alpha, "flexBicepsChallenge", "connectTheNeuronsChallenge",
        #                    "listenUpChallenge", "feelTheSensationChallenge",
        #                    actLength=False, wrongAct=False, infoAct=False,
        #                    pauseAct=False, mcScore=False, mcProblem=False, challengeAmt=True, preScore=False,
        #                    lowPreUsers=False,
        #                    preScoreMedian=False)
        # lassoRegressionBin(Dataset, alpha, "flexBicepsChallenge", "connectTheNeuronsChallenge",
        #                    "listenUpChallenge", "feelTheSensationChallenge",
        #                    actLength=False, wrongAct=False, infoAct=False,
        #                    pauseAct=False, mcScore=False, mcProblem=False, challengeAmt=False, preScore=True,
        #                    lowPreUsers=False,
        #                    preScoreMedian=False)
        # lassoRegressionBin(Dataset, alpha, "flexBicepsChallenge", "connectTheNeuronsChallenge",
        #                    "listenUpChallenge", "feelTheSensationChallenge",
        #                    actLength=False, wrongAct=False, infoAct=False,
        #                    pauseAct=False, mcScore=False, mcProblem=False, challengeAmt=False, preScore=False,
        #                    lowPreUsers=False,
        #                    preScoreMedian=True)


        # lassoRegressionBin(Dataset, alpha, "flexBicepsChallenge", "connectTheNeuronsChallenge",
        #                    "listenUpChallenge", "feelTheSensationChallenge",
        #                    actLength=True, wrongAct=True, infoAct=True,
        #                    pauseAct=True, mcScore=True, mcProblem=True, challengeAmt=True, preScore=True,
        #                    lowPreUsers=False,
        #                    preScoreMedian=True)
        #
        #
        # alpha += 0.1

    print("--- %s seconds ---" % (time.time() - start_time))