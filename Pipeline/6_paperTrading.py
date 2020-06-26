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
    #wont use but if we do, this is wrong
    #just do buy max
    def buyStock(self, transaction):
        while(self.validBuyTransaction(transaction) == False):
            transaction.oneLessShare()
        if (transaction.getNumShares() > 0):
            self.transactions.append(transaction.listTransaction())
            self.numSharesOwned += transaction.getNumShares()
            self.balance -= transaction.getTotalPrice()

    #so just sell max instead
    #sellStock will attempt to sell the amount of stock A specified and if the Portfolio doesn't at least have the # of stock specified, will sell as much of stock A as possible
    def sellStock(self, transaction):
        while (transaction.getNumShares() > self.numSharesOwned):
            transaction.oneLessShare()
        if (transaction.getNumShares() > 0):
            self.numSharesOwned -= transaction.getNumShares()
            self.balance += transaction.getTotalPrice()
            self.transactions.append(transaction.listTransaction())

    def buyMax(self, price, date, time):
        startingShares = self.numSharesOwned
        startingBalance = self.balance
        maxSharesCanBuy = math.floor(self.balance / price)
        if (maxSharesCanBuy != 0):
            self.balance -= maxSharesCanBuy * price
            self.numSharesOwned += maxSharesCanBuy
            buyTrans = Transaction(self.ticker, price, date, time, "Buy", startingShares, self.numSharesOwned, startingBalance, self.balance)
            self.transactions.append(buyTrans.listTransaction())

    def sellMax(self, price, date, time): #will take any transaction and make it sell shares
        startingShares = self.numSharesOwned
        startingBalance = self.balance
        if (self.numSharesOwned != 0):
            self.balance += self.numSharesOwned * price
            self.numSharesOwned = 0
            sellTrans = Transaction(self.ticker, price, date, time, "Sell", startingShares, self.numSharesOwned, startingBalance, self.balance)
            self.transactions.append(sellTrans.listTransaction())

    def getTransactions(self):
        return self.transactions

    #displayTransactions will print out all the transactions in the Portfolio within the timeframe specified
    def displayTransactions(self, startDate, endDate): #startDate and endDate should be date objects
        if (endDate < startDate):
            print("Stop trolling bro: can't have end date before start date")
            return
        for trans in self.transactions:
            if (trans.getDate() < startDate):
                continue
            if (trans.getDate() >= endDate):
                break
            print(trans)
            print()

    #displayAllTransactions will print out all the transactions in the Portfolio
    def displayAllTransactions(self):
        for trans in self.transactions:
            print(trans)


    #validBuyTransaction will return whether we can complete a buy transaction
    def validBuyTransaction(self, transaction):
        if (transaction.getTotalPrice() <= self.balance):
            return True
        else:
            return False

    def getTickerName(self):
        return self.ticker

class Transaction:

    def __init__(self, ticker, price, date, time, transactionType, sharesBefore, sharesAfter, balanceBefore, balanceAfter):
        self.ticker = ticker
        self.price = price
        self.date = date
        self.time = time # time will be either "open" or "close" to determine if we are buying/selling at open or close
        self.transactionType = transactionType # transactionType will be a string "Buy" or "Sell"
        self.sharesBefore = sharesBefore
        self.sharesAfter = sharesAfter
        self.balanceBefore = balanceBefore
        self.balanceAfter = balanceAfter

        #These are calculated
        self.numShares = abs(sharesAfter - sharesBefore)

        if (transactionType == "Buy"):
            self.transPrice = -1*(self.price * self.numShares)
        else:
            self.transPrice = self.price * self.numShares

    def listTransaction(self):
        return [str(self.date), str(self.time), str(self.ticker), str(self.price), str(self.transactionType), str(self.sharesBefore), str(self.sharesAfter), str(self.balanceBefore), str(self.balanceAfter)]

    #getDate will return the date of the transaction
    def getDate(self):
        return self.date

    #getTransactionPrice will return the total price of the transaction
    def getTransactionPrice(self):
        return self.transPrice


    #getNumShares will return the number of shares specified in this transaction
    def getNumShares(self):
        return self.numShares

    #getTransactionType will return whether the transaction is a buy or a sell
    def getTransactionType(self):
        return self.transactionType

    #oneLessShare allows us to decrease the number of shares by 1 and is used when trying to buy or sell as much as possible of stock A
    def oneLessShare(self):
        self.numShares -=1
        self.totalPrice = self.shareCost * self.numShares

    def oneMoreShare(self):
        self.numShares += 1
        self.totalPrice = self.shareCost * self.numShares

def main():
    #for friday: have control algorithm run, and have algorithm #1 run, and output it somehow with a csv
    #titleTicker = (lower(_configKeys.YVALUETICKER)).title() # ex: yvalueticker = "GOLD", titleTicker = "Gold"
    startBalance = 1000
    window_length = _configKeys.WINDOW_LENGTH
    thresholdPortfolio = Portfolio(startBalance, _configKeys.YVALUETICKER)
    controlPortfolio = Portfolio(startBalance, _configKeys.YVALUETICKER)

    testingDF = pd.read_csv(os.path.join(Path(_configKeys.TESTING_RESULTS_FOLDER), _configKeys.YVALUETICKER + "0.3_alpha13_beta_test_results.csv"))
    dataDF = pd.read_csv(os.path.join(Path(_configKeys.DATA_FOLDER), _configKeys.YVALUETICKER+".csv"))
    weeksDict = daysInWeekDict(dataDF)

    #algorithms being run on portfolios
    runControl(controlPortfolio, testingDF, dataDF, weeksDict)
    runThresh(thresholdPortfolio, testingDF, dataDF, weeksDict, startBalance)

    threshTransList = thresholdPortfolio.getTransactions()
    threshDF = pd.DataFrame(columns = ["Date", "Time", "Ticker", "Price", "Transaction Type", "Shares before", "Shares After", "Balance Before", "Balance After"])
    for i in range(len(threshTransList)):
        threshDF.append(threshTransList[i])
    threshDF.to_csv(os.path.join(Path(_configKeys.PAPER_RESULTS_FOLDER, _configKeys.YVALUETICKER+"_threshold.csv")))

    controlTransList = controlPortfolio.getTransactions()
    controlDF = pd.DataFrame(columns = ["Date", "Time", "Ticker", "Price", "Transaction Type", "Shares before", "Shares After", "Balance Before", "Balance After"])
    for i in range(len(controlTransList)):
        controlDF.append(threshTransList[i])
    controlDF.to_csv(os.path.join(Path(_configKeys.PAPER_RESULTS_FOLDER, _configKeys.YVALUETICKER+"_control.csv")))

    #def __init__(self, ticker, price, date, time, transactionType, sharesBefore, sharesAfter, balanceBefore, balanceAfter):

    #(a, b) for (a, b) in range(len(thresholdPortfolio.getTransactions())):




def runControl(controlPortfolio, testingDF, dataDF, weeksDict):
    algorithm_Control(controlPortfolio, testingDF, dataDF, weeksDict)
    print ("Control Profit: " + str(controlPortfolio.getTotalProfit()))

def runThresh(thresholdPortfolio, testingDF, dataDF, weeksDict, startBalance):
    #Threshold Testing:
    threshold = 0.051 # we can start with this arbitrary threshold that is close to the most profitable threshold
    algorithm_ApproachThreshold(thresholdPortfolio, testingDF, dataDF, weeksDict, threshold, False)
    #thresholdPortfolio.displayAllTransactions()
    print("Threshold Algorithm Profit: " + str(thresholdPortfolio.getTotalProfit()))

#Take in day dataDF
#Will return a dictionary with the weekly dates as keys and a DataFrame of daily values during that week
def daysInWeekDict(dataDF):
    retDict = {}
    firstIndex = datetime.datetime.strptime(_configKeys.FIRSTINDEX, '%Y-%m-%d')
    lastIndex = datetime.datetime.strptime(_configKeys.LASTINDEX, '%Y-%m-%d')
    difference = lastIndex.date() - firstIndex.date()
    #we start at the first week and will update date to match what we want (first day of week)
    date = firstIndex.date()
    numWeeks = math.ceil(difference.days/7)

    for i in range(numWeeks):
        #print(date.strftime("%Y-%m-%d"))

        retDict[date.strftime("%Y-%m-%d")] = pd.DataFrame(columns = dataDF.columns)
        dates = [] # keeps track of valid dates (String) for a given week
        for j in range(6):
            currentDate = date + datetime.timedelta(days=j)
            if currentDate.strftime("%Y-%m-%d") in list(dataDF["Date"].values):
                dates.append(currentDate.strftime("%Y-%m-%d"))
                #print(currentDate.strftime("%Y-%m-%d"))

        for day in dates:
            dayRow = dataDF.loc[dataDF['Date'] == day]
            retDict[date.strftime("%Y-%m-%d")] = retDict[date.strftime("%Y-%m-%d")].append(dayRow)
        date += datetime.timedelta(days=7)

    return retDict

def algorithm_Control(portfolio, testingDF, dataDF, weekDict):
    #firstIndex = datetime.datetime.strptime(_configKeys.FIRSTINDEX, '%Y-%m-%d')
    #lastIndex = datetime.datetime.strptime(_configKeys.LASTINDEX, '%Y-%m-%d')
    firstDate = list(testingDF["Date"].values)[0] #Set this to be the first day of the testingDF
    lastDate = list(dataDF["Date"].values)[-1]
    #date = firstDate
    boughtAtStart = False
    #simply grab value from first date and last date and buy/sell at those times respectively
    for week in list(testingDF["Date"].values):
        weekDF = weekDict[week]
        for day in list(weekDF["Date"].values):
            if week == firstDate and boughtAtStart == False:
                dayRowIndex = dataDF.index[dataDF['Date'] == day][0]
                price = dataDF.at[dayRowIndex, "Open"]
                portfolio.buyMax(price, day, "Open")
                boughtAtStart = True
            if day == lastDate:
                dayRowIndex = dataDF.index[dataDF['Date'] == day][0]
                price = dataDF.at[dayRowIndex, "Close"]
                portfolio.sellMax(price, day, "Close")

'''
CAN IGNORE-- we may want to still consider a changing the num shares we buy
Smarter Threshold algorithm ideas:
    1. More speciic buying / selling (E.g. if a stock is within the predicted high minus the threshold, sell x# of stock, if it's within the predicted high + threshold, sell x#*2 of stock)
    2. Looking at past week actuals versus current week predicted
        a. (E.g Actual high of first week of trading was higher than current week of trading high, therefore we do no selling no matter what)
'''

#portfolio is a portfolio, price is the close price of a week
def bestThreshold(testingDF, dataDF,weekDict, date, price, threshold):
    window_length = _configKeys.WINDOW_LENGTH
    #date must be the sunday after the trading week
    startThres = int(price*.016*1000) #We picked .016 and .022 as the starting percentages of any stock price based on their initial success with gold specifically
    endThres = int(price*.022*1000)
    firstDateString = list(testingDF["Date"].values)[0] #Set this to be the first day of the testingDF
    firstDate = datetime.datetime.strptime(firstDateString, '%Y-%m-%d')
    difference = date.date() - firstDate.date() #difference
    weeksElapsed = math.ceil(difference.days/7) # weeks elapsed is difference in days

    portfolio_profit_list = []
    thresholdList = []

    testingIntervalDF = pd.DataFrame(columns = testingDF.columns)

    if weeksElapsed >= window_length:
        iterDate = date - timedelta(days=window_length*7) # using this to iter through currentDate - window_length to currentDate

        for i in range(window_length):
            weekRow = testingDF.loc[testingDF['Date'] == datetime.datetime.strftime(iterDate, '%Y-%m-%d')]
            testingIntervalDF = testingIntervalDF.append(weekRow)
            iterDate = iterDate + timedelta(days=7)

        for i in range(startThres, endThres):
            #portfolio = MainThresholdPortfolio
            #algorithm_ApproachThreshold(portfolio, testingDF(from current week - WINDOW_LENGTH), dataDF, weekDict, i / 1000, numShares, price)

            portfolio = Portfolio(1000, _configKeys.YVALUETICKER)
            algorithm_ApproachThreshold(portfolio, testingIntervalDF, dataDF, weekDict, i / 1000, True)
            portfolio_profit_list.append(portfolio.getTotalProfit())
            thresholdList.append(i / 1000)

        return thresholdList[portfolio_profit_list.index(max(portfolio_profit_list))]
    else:
        return threshold


def algorithm_ApproachThreshold(portfolio, testingDF, dataDF, weekDict, threshold, isTest):
    #titleTicker = (lower(_configKeys.YVALUETICKER)).title() # ex: yvalueticker = "GOLD", titleTicker = "Gold"

    ticker = portfolio.getTickerName()
    endWeekPrice = 0 # initializing to keep track so we can feed this to best threshold algorithm to test thresholds appropriately

    firstDate = list(testingDF["Date"].values)[0] #Set this to be the first day of the testingDF
    lastDate = list(dataDF["Date"].values)[-1]
    #list of features we want to track

    featureList = ["Gold" + "_Low_average_Predicted", "Gold" +"_High_average_Predicted"] # make it uniform -- we want gold ticker to be capitalized
    #for each week in our testingDF, get desired predictedFEATURE(s) from the list
    for week in list(testingDF["Date"].values):
        #Get the index of the current week within the testing dataframe
        weekRowIndex = testingDF.index[testingDF['Date'] == week][0]

        #Find the low_average_predicted and high_max_predicted for that week
        lowMinP = testingDF.at[weekRowIndex, featureList[0]]
        highMaxP = testingDF.at[weekRowIndex, featureList[1]]

        #Get the dataframe holding the weekly values for this week
        weekDF = weekDict[week]
        #daytradeCount holds the number of day trades we've done in a week to make sure we don't day trade too many times
        daytradeCount = 0

        #For each daily value in the weekDF
        for day in list(weekDF["Date"].values):
            #run a list on the last 5 valid dates? -- then
            dayTradeDateLists = []
            #3 counters that keep track of day trades -- maybe an int that stores the number of days since the first day trade and subtract 1 every day

            #Get the index of the current day within the weekly dataframe
            dayRowIndex = dataDF.index[dataDF['Date'] == day][0]

            #we approximate the 9:30:01AM EST (when we would actually buy/sell) value of the stock with the close price (In practice, we would simply get the stock price at that point and run it through our buy / sell algorithm)
            nearOpen = dataDF.at[dayRowIndex, "Open"]
            #we approximate the 3:59:59PM EST (when we would actually buy/sell) value of the stock with the close price (In practice, we would simply get the stock price at that point and run it through our buy / sell algorithm)
            nearClose = dataDF.at[dayRowIndex, "Close"]

            endWeekPrice = nearClose
            #isLastDay checks to see if we are on the last trading day of the week so that we only sell at the open or close
            isLastDay = False
            transMadeToday = False
            #Checks to see if we are on the last day of the week
            if day == list(weekDF["Date"].values)[-1]:
                isLastDay = True
            #Allows for 3 day trade days at the most and will run on every trading day except last (on last day, we only sell)
            if (daytradeCount < 3 and isLastDay == False):
                #Open threshold checks
                if (nearOpen < lowMinP + threshold):
                    portfolio.buyMax(nearOpen, day, "Open")
                    transMadeToday = True
                elif (nearOpen > highMaxP - threshold):
                    portfolio.sellMax(nearOpen, day, "Open")
                    transMadeToday = True

                #Close thresholds checks
                #if we have not max day traded, then we do this
                if daytradeCount < 3: # this is redundant - what do we really want to check?

                    if (nearClose < lowMinP + threshold):
                        if transMadeToday:
                            daytradeCount += 1
                        portfolio.buyMax(nearClose, day, "Close")
                    elif (nearClose > highMaxP - threshold):
                        portfolio.sellMax(nearClose, day, "Close")
                        if transMadeToday:
                            daytradeCount += 1
                #if we havent trading earlier in the day and we cant day trade anymore, we run algotrader
                elif (transMadeToday == False):

                    if (nearClose < lowMinP + threshold):
                        portfolio.buyMax(nearClose, day, "Close")
                    elif (nearClose > highMaxP - threshold):
                        portfolio.sellMax(nearClose, day, "Close")
            #If we are in the last day of the trading week, sell all we have at open or close
            if (isLastDay):
                    if (nearOpen > highMaxP - threshold):
                        portfolio.sellMax(nearOpen, day, "Open")
                    else:
                        portfolio.sellMax(nearClose, day, "Close")

        inputWeek = datetime.datetime.strptime(week, "%Y-%m-%d")
        if isTest == False:
            threshold = bestThreshold(testingDF, dataDF, weekDict, inputWeek, endWeekPrice, threshold)

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
#wireframe next week: outline of app with buttons but no functionality required

main()
