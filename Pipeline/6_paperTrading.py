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

class Portfolio:
    def __init__(self, balance, ticker):
        self.balance = balance
        self.transactions = []
        self.numSharesOwned = 0
        self.initialBalance = balance
        #in this simplified version of a portfolio, we will only be able to purchase a single stock
        self.ticker = ticker

    #addBalance will increase the Portfolio balance by add
    def addBalance(self, add):
        self.balance += add

    #getTotalProfit will return the total profit a Portfolio has gained
    def getTotalProfit(self):
        return self.balance - self.initialBalance

    #getBalance will return the Portfolio's balance
    def getBalance(self):
        return self.balance

    #buyStock will attempt to buy the amount of stock A specified and if the Portfolio doesn't have enough money, will buy as much of stock A as possible
    def buyStock(self, transaction):
        while(self.validTransaction(transaction) == False):
            transaction.oneLessShare()
        self.transactions.append(transaction)
        self.numSharesOwned += transaction.getNumShares()
        self.balance -= transaction.getTotalPrice()

    #sellStock will attempt to sell the amount of stock A specified and if the Portfolio doesn't at least have the # of stock specified, will sell as much of stock A as possible
    def sellStock(self, transaction):
        while (transaction.getNumShares() > self.numSharesOwned):
            transaction.oneLessShare()
        self.transactions.append(transaction)
        self.numSharesOwned -= transaction.getNumShares()
        self.balance += transaction.getTotalPrice()

    #validTransaction will return whether we can complete a transaction
    def validTransaction(self, transaction):
        if (transaction.getTotalPrice() <= self.balance):
            return True
        else:
            return False

    def getTickerName(self):
        return self.ticker

class Transaction:
    def __init__(self, ticker, shareCost, numShares, date, transactionType, time):
        self.ticker = ticker
        self.shareCost = shareCost
        self.numShares = numShares
        self.date = date
        self.transactionType = transactionType # transactionType will be a string "Buy" or "Sell"
        self.time = time # time will be either "open" or "close" to determine if we are buying/selling at open or close
        self.totalPrice = transaction.shareCost * transaction.numShares


    #getTotalPrice will return the total price of the transaction
    def getTotalPrice(self):
        return self.totalPrice

    #getNumShares will return the number of shares specified in this transaction
    def getNumShares(self):
        return self.numShares

    #getTransactionType will return whether the transaction is a buy or a sell
    def getTransactionType(self):
        return self.transactionType

    #oneLessShare allows us to decrease the number of shares by 1 and is used when trying to buy or sell as much as possible of stock A
    def oneLessShare(self):
        self.numShares -=1
        self.totalPrice = transaction.shareCost * transaction.numShares

def main():
    #for friday: have control algorithm run, and have algorithm #1 run, and output it somehow with a csv
    startBalance = 1000
    window_length = _configKeys.WINDOW_LENGTH
    myPortfolio = Portfolio(startBalance, _configKeys.YVALUETICKER)
    controlPortfolio = Portfolio(startBalance, _configKeys.YVALUETICKER)

    meanErrorList = [] #meanErrorList is a list of tuples equal to a (featureName, meanErrorForFeatureName)

    testingDF = pd.read_csv(os.path.join(Path(_configKeys.TESTING_RESULTS_FOLDER), "GOLD0.3_alpha13_beta_test_results.csv"))
    dataDF = pd.read_csv(os.path.join(Path(_configKeys.DATA_FOLDER), "GOLD.csv"))
    weeksDict = daysInWeekDict(dataDF)


    for i in range(2, len(testingDF.columns)-1):
        col = testingDF.columns[i]
        nextCol = testingDF.columns[i+1]

        if "Predicted" in col:
            colName = col.split("_")[:-1]
            colName = str(colName[0]) + "_" + str(colName[1]) + "_" + str(colName[2])

            predList = list(testingDF[col].values)
            actList = list(testingDF[nextCol].values)

            meanError = calculateMeanError(actList, predList)
            meanErrorList.append((colName, meanError))

    print(meanErrorList)

    lassoProfit = myPortfolio.getTotalProfit()
    controlProfit = controlPortfolio.getTotalProfit()
    print ("Lasso Profit: " + str(lassoProfit))
    print ("Control Profit: " + str(controlProfit))


#Take in day dataDF
#Will return a dicionary with the weekly dates as keys and a DataFrame of daily values during that week

def daysInWeekDict(dataDF):
    #TODO: make this return what we want (a DataFrame with all trading days for a specific week)
    retDict = {}
    firstIndex = datetime.datetime.strptime(_configKeys.STARTPULL, '%d/%m/%Y')
    lastIndex = datetime.datetime.strptime(_configKeys.ENDPULL, '%d/%m/%Y')
    difference = lastIndex.date() - firstIndex.date()
    #we start at the first week and will update currentDate to match what we want (first day of week)
    date = firstIndex.date() + timedelta(days=1)
    placeholderDict = {} # will contain weekly values in dict

    numWeeks = int(difference.days/7)
    print(numWeeks)

    for i in range(numWeeks):
        retDict[date] = pd.DataFrame(columns = dataDF.columns)
        date += datetime.timedelta(days=7)
        dates = [] # keeps track of valid dates for a given week
        for i in range(5):

            currentDate = date + datetime.timedelta(days=i)
            if str(currentDate) in list(dataDF["Date"].values):
                dates.append(str(currentDate))
                print(str(currentDate))

        print(dataDF.loc[dates])
        quit()
        retDict[date] += dataDF.loc[dates]

    return retDict
'''
    rows = dataDF.iterrows()
    print(rows[1])
    index = 0
    for i in range(int(difference.days)):
        date += datetime.timedelta(days=1)
        if date in list(dataDF["Date"].values):
            #if this is first day of new week
            if int((date - firstIndex.date()).days) == 0:
                placeholderDict[currentDate] = dataDF.iloc[index]
                index += 1

            #if difference in days is less than 5 - within same week
            elif int((date - firstIndex.date()).days) < 5:
                prevDF = placeholderDict[currentDate]

'''




    #reference_df = pd.read_csv("1successfulPulls.csv", low_memory=False)
    #weekDF = []
    #for date in list(df["Date"].values):

    #return daysInWeekDict = GetWeekDictionary(dataDF, asset_class_has_volume)



def algorithm_MeanError(portfolio, testingDF, dataDF, weekDict):
    #only use error for a specific window
    ticker = portfolio.getTickerName()
    for week in list(testingDF["Date"].values):
        #grabbing daily values for this week (we have the daily data in a DataFrame with 5 values)
        daysDF = weekDict[week] #TODO: make into a dataframe with [Date, Open High, Low, Close, Volume] as the headers

        #these are values we want to compare each daily ACTUAL value (Formatting -- maybe just make every word start with a capital Letter -- bring up in meeting)
        list(testingDF[ticker + "_Open_max_Predicted"].values)

        #for day in list(daysDF["Date"].values):


#idea- buy, and only sell if it goes above a predicted max

def algorithm_ApproachThreshold(portfolio, testingDF, dataDF, weekDict):
    ticker = portfolio.getTickerName()
    #Idea: ex- If the actual average High of first week of trading is greater than the predicted average high of second week, we do no buying at all no matter what

    #list of features we want to track
    featureList = [ticker+"_Open_max_Predicted",ticker+"_High_volatility_Predicted", ticker+"_High_max_Predicted"]

    #for each week in our testingDF, get desired predictedFEATURE(s) from the list
    for week in list(testingDF["Date"].values):
        weekDF = weekDict[week]
        #for each daily value in the weekDF
        for day in list(weekDF["Date"].values):
            open = weekDF.iloc()
            close = weekDF

            #compare if that daily value is withing predicted value within threshold
                #if "low" in predictedFeature - think about this
                    #buy shares beginning of next trading day
                #if "high" in predictedFeature
                    #sell shares beginning of next trading day


#calculateMeanError will find the mean error between two lists: predictionList: the predicted data for a feature, actualList: the actual data for a feature
def calculateMeanError(actualList, predictionList):
    #maybe lets actually just do sqrt(variance)??
    totalMean = 0
    for i in range(len(actualList)):
        totalMean += abs(actualList[i] - predictionList[i])
    return totalMean/len(actualList)

'''
Using multi armed bandit on all our algorithms to figure out which has least regret
Possible algorithms: Should work with different featureNames in order to see which feature is the best indicator
1. Use predictions for next week and when next week comes, compare important actual values to predicted weekly values and when actual approaches predicted, sell / buy
2. Mean Error as decider of how many shares to buy
3. Control algorithm: Buy beginning of the week, sell at the end
'''


def GetWeekDictionary(assetDF, include_volume):

    '''
    This piece of code breaks up the daily csv into weeks
    '''

    startBinDatetime, endBinDatetime = datetime.datetime.strptime(_configKeys.STARTPULL, '%d/%m/%Y'), datetime.datetime.strptime(_configKeys.ENDPULL, '%d/%m/%Y')

    countDatetime = startBinDatetime
    #bins = [] might not need this anymore
    datetimeBin = {}

    while (countDatetime < endBinDatetime): # while the count time is not at the last week in the sequence
        datetimeBin[countDatetime] = []
        #bins.append(datetimeBin) we might not need this code anymore
        countDatetime = countDatetime + timedelta(days=7)


    #This first puts the y value into the bins list. This is to give us easy access when trying to move it to the yValues list

    assetWeek = []
    currentBinDate = startBinDatetime

    for ind in assetDF.index:

        # Current date for stock is past current bin.
        if (datetime.datetime.strptime(assetDF['Date'][ind], '%Y-%m-%d') - currentBinDate).days > 7:
            datetimeBin[currentBinDate] = assetWeek
            currentBinDate = currentBinDate + timedelta(days=7)
            if include_volume == True:
                assetWeek = [[datetime.datetime.strptime(assetDF['Date'][ind], '%Y-%m-%d'), assetDF['Open'][ind], assetDF['High'][ind], assetDF['Low'][ind], assetDF['Close'][ind], assetDF['Volume'][ind]]]
            else:
                assetWeek = [[datetime.datetime.strptime(assetDF['Date'][ind], '%Y-%m-%d'), assetDF['Open'][ind], assetDF['High'][ind], assetDF['Low'][ind], assetDF['Close'][ind]]]
        else:
            if include_volume == True:
                assetWeek.append([datetime.datetime.strptime(assetDF['Date'][ind], '%Y-%m-%d'), assetDF['Open'][ind], assetDF['High'][ind], assetDF['Low'][ind], assetDF['Close'][ind], assetDF['Volume'][ind]])
            else:
                assetWeek = [[datetime.datetime.strptime(assetDF['Date'][ind], '%Y-%m-%d'), assetDF['Open'][ind], assetDF['High'][ind], assetDF['Low'][ind], assetDF['Close'][ind]]]

    # We have to do this one more time to get the values from the last week
    datetimeBin[currentBinDate] = assetWeek

    return datetimeBin
#Inital: Both Control Reward and Lasso Reward start with a set amount of dollar

#Goals: What happens if we bought the stock and just held on to it? (Control Reward)

#Goals: How do we make use of the lasso file to tell us when to buy and sell? (Lasso Reward)
# reward = difference between actual values -- choices are only influenced by predicted values however
# d (what if we buy at start of week and sell at end of week??? - think about using data in smart ways to make cool algoriths) -- open 1Data folder and use datetime to get the specific open
# possible alg for paper trading:
# take mean error for our prediction (ex: open average) -- if mean error is 50 cents, and our prediction is above a 50 cent increase, then buy, else don't -- need to "wait" 10 week_bin_list
# if mean error > predicted increase, then don't buy

# also maybe consider volatility







#wireframe next week: outline of app with buttons but no functionality required

main()
