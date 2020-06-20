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
        if (transaction.getTotalPrice() <= balance):
            return True
        else:
            return False

    def getTickerName(self):
        return self.ticker

class Transaction:
    def __init__(self, ticker, shareCost, numShares, date, transactionType):
        self.ticker = ticker
        self.shareCost = shareCost
        self.numShares = numShares
        self.date = date
        self.transactionType = transactionType # transactionType will be a string "Buy" or "Sell"
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

    #oneLessShare allows us to decrease the number of shares wanted and is used when trying to buy or sell as much as possible of stock A
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
    reference_df = pd.read_csv("1successfulPulls.csv", low_memory=False)
    asset_class_has_volume = False
    if asset_class == "Stock" or asset_class == "Commodity":
        asset_class_has_volume = True
    return daysInWeekDict = GetWeekDictionary(dataDF, asset_class_has_volume)



def algorithm_MeanError(portfolio, testingDF, dataDF, weekDict):
    ticker = portfolio.getTickerName()
    for week in list(testingDF["Date"].values):
        #grabbing daily values for this week (we have the daily data in a DataFrame with 5 values)
        daysDF = weekDict[week] #TODO: make into a dataframe with [Date, Open High, Low, Close, Volume] as the headers

        #these are values we want to compare each daily ACTUAL value (Formatting -- maybe just make every word start with a capital Letter -- bring up in meeting)
        list(testingDF[ticker + "_Open_max_Predicted"].values)

        for day in list(daysDF["Date"].values):


def algorithm_HighLowThreshold(portfolio, testingDF, dataDF, weekDict):
    ticker = portfolio.getTickerName()
    #Idea: ex- If the actual average High of first week of trading is greater than the predicted average high of second week, we do no buying at all no matter what

    #list of features we want to track
    featureList = [ticker+"_Open_max_Predicted",ticker+"_High_volatility_Predicted", ticker+"_High_max_Predicted"]

    #for each week in our testingDF, get desired predictedFEATURE(s) from the list
    for week in list(testingDF["Date"].values):
        weekDF = weekDict[week]
        #for each daily value in the weekDF
        for day in list(weekDF["Date"].values):
            open = weekDF[""]
            close =

            #compare if that daily value is withing predicted value within threshold
                #if "low" in predictedFeature - think about this
                    #buy shares beginning of next trading day
                #if "high" in predictedFeature
                    #sell shares beginning of next trading day
    pass()

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


#Inital: Both Control Reward and Lasso Reward start with a set amount of dollar

#Goals: What happens if we bought the stock and just held on to it? (Control Reward)

#Goals: How do we make use of the lasso file to tell us when to buy and sell? (Lasso Reward)
# reward = difference between actual values -- choices are only influenced by predicted values however
# d (what if we buy at start of week and sell at end of week??? - think about using data in smart ways to make cool algoriths) -- open 1Data folder and use datetime to get the specific open
# possible alg for paper trading:
# take mean error for our prediction (ex: open average) -- if mean error is 50 cents, and our prediction is above a 50 cent increase, then buy, else don't -- need to "wait" 10 week_bin_list
# if mean error > predicted increase, then don't buy

# also maybe consider volatility
main()
