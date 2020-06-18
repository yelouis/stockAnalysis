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
    def __init__(self, balance):
        self.balance = balance
        self.transactions = []
        self.numSharesOwned = 0
        self.initialBalance = balance

    #addBalance will increase the Portfolio balance by add
    def addBalance(self, add):
        self.balance += add

    def getTotalProfit(self):
        return self.balance - self.initialBalance

    #getBalance will return the Portfolio's balance
    def getBalance(self):
        return self.balance

    #buyStock will attempt to buy the amount of stock A specified and if the Portfolio doesn't have enough money, will buy as much of stock A as possible
    def buyStock(self, transaction):
        while(!self.validTransaction(transaction)):
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

class Transaction:
    def __init__(self, ticker, shareCost, numShares, totalPrice, date, transactionType):
        self.ticker = ticker
        self.shareCost = shareCost
        self.numShares = numShares
        self.totalPrice = totalPrice
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
    startBalance = 1000

    myPortfolio = Portfolio(startBalance)
    controlPortfolio = Portfolio(startBalance)





    lassoProfit = myPortfolio.getTotalProfit()
    controlProfit = controlPortfolio.getTotalProfit()
    print ("Lasso Profit: " + str(lassoProfit))
    print ("Control Profit: " + str(controlProfit))

#calculateMeanError will find the mean error
def calculateMeanError(actualList, predictionList):
    totalMean = 0
    for i in range(len(actualList)):
        totalMean += abs(actualList[i] - predictionList[i])
    return totalMean/len(actualList)

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
