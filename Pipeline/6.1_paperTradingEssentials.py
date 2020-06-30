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

    #getTickerName returns the name of the stock tracked in this Portfolio
    def getTickerName(self):
        return self.ticker

    #addBalance will increase the Portfolio balance by add
    def addBalance(self, add):
        self.balance += add

    #getBalance will return the Portfolio's balance
    def getBalance(self):
        return self.balance

    #getTotalProfit will return the total profit a Portfolio has gained
    def getTotalProfit(self):
        return self.balance - self.initialBalance

    #buyStock will attempt to buy the amount of stock A specified and if the Portfolio doesn't have enough money, will buy as much of stock A as possible
    #wont use but if we do, this is wrong
    def buyStock(self, transaction):
        while(self.validBuyTransaction(transaction) == False):
            transaction.oneLessShare()
        if (transaction.getNumShares() > 0):
            self.transactions.append((transaction, (self.numSharesOwned, self.balance)))
            self.numSharesOwned += transaction.getNumShares()
            self.balance -= transaction.getTotalPrice()

    #sellStock will attempt to sell the amount of stock A specified and if the Portfolio doesn't at least have the # of stock specified, will sell as much of stock A as possible
    def sellStock(self, transaction):
        while (transaction.getNumShares() > self.numSharesOwned):
            transaction.oneLessShare()
        if (transaction.getNumShares() > 0):
            self.numSharesOwned -= transaction.getNumShares()
            self.balance += transaction.getTotalPrice()
            self.transactions.append((transaction, (self.numSharesOwned, self.balance)))

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
        return [str(self.ticker), str(self.price), str(self.date), str(self.time), str(self.transactionType), str(self.sharesBefore), str(self.sharesAfter), str(self.balanceBefore), str(self.balanceAfter)]

    #getDate will return the date of the transaction
    def getDate(self):
        return self.date

    #getTotalPrice will return the total price of the transaction
    def getTransactionPrice(self):
        return -self.transPrice

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

    #oneMoreShare allows us to increase the number of shares by 1 and is used when trying to buy or sell as much as possible of stock A
    def oneMoreShare(self):
        self.numShares += 1
        self.totalPrice = self.shareCost * self.numShares



def main():
    #Setting starting balance for portfolio and window_length
    startBalance = 1000
    window_length = _configKeys.WINDOW_LENGTH

    #Algorithm portfolio initializations
    thresholdPortfolio = Portfolio(startBalance, _configKeys.YVALUETICKER)
    controlPortfolio = Portfolio(startBalance, _configKeys.YVALUETICKER)

    #Initializing data structures
    testingDF = pd.read_csv(os.path.join(Path(_configKeys.TESTING_RESULTS_FOLDER), _configKeys.YVALUETICKER + "0.3_alpha13_beta_test_results.csv"))
    dataDF = pd.read_csv(os.path.join(Path(_configKeys.DATA_FOLDER), _configKeys.YVALUETICKER + ".csv"))
    weeksDict = daysInWeekDict(dataDF)

    #Algorithms being run on portfolios
    runControl(controlPortfolio, testingDF, dataDF, weeksDict)
    runThresh(thresholdPortfolio, testingDF, dataDF, weeksDict, startBalance)


    threshTransList = thresholdPortfolio.getTransactions()
    threshDF = pd.DataFrame(columns = ["Ticker", "Price", "Date", "Time", "Transaction Type", "Shares before", "Shares After", "Balance Before", "Balance After"])
    for i in range(len(threshTransList)):
        transaction = pd.Series(threshTransList[i], index = threshDF.columns)
        threshDF = threshDF.append(transaction, ignore_index = True)
    threshDF.to_csv(os.path.join(Path(_configKeys.PAPER_RESULTS_FOLDER), _configKeys.YVALUETICKER+"_thresholdByEstimate.csv"))

    controlTransList = controlPortfolio.getTransactions()
    controlDF = pd.DataFrame(columns = ["Date", "Time", "Ticker", "Price", "Transaction Type", "Shares before", "Shares After", "Balance Before", "Balance After"])
    for i in range(len(controlTransList)):
        transaction = pd.Series(controlTransList[i], index = controlDF.columns)
        controlDF = controlDF.append(transaction, ignore_index=True)
    controlDF.to_csv(os.path.join(Path(_configKeys.PAPER_RESULTS_FOLDER), _configKeys.YVALUETICKER+"_control.csv"))

#Runs the control algorithm and prints the profit achieved
def runControl(controlPortfolio, testingDF, dataDF, weeksDict):
    algorithm_Control(controlPortfolio, testingDF, dataDF, weeksDict)
    print ("Control Profit: " + str(controlPortfolio.getTotalProfit()))

def runThresh(thresholdPortfolio, testingDF, dataDF, weeksDict, startBalance):
    #Threshold Testing:
    day = list(dataDF["Date"].values)[-1]
    dayRowIndex = dataDF.index[dataDF['Date'] == day][0]
    price = dataDF.at[dayRowIndex, "Close"]
    algorithm_Threshold(thresholdPortfolio, testingDF, dataDF, weeksDict, 0, False)
    print("Threshold Algorithm Profit: " + str(thresholdPortfolio.getTotalProfit()))

def daysInWeekDict(dataDF):
    retDict = {}
    firstIndex = datetime.datetime.strptime(_configKeys.FIRSTINDEX, '%Y-%m-%d')
    lastIndex = datetime.datetime.strptime(_configKeys.LASTINDEX, '%Y-%m-%d')
    difference = lastIndex.date() - firstIndex.date()
    #we start at the first week and will update date to match what we want (first day of week)
    date = firstIndex.date() #+ timedelta(days=1) put this back in when louis/cole change 1Data
    numWeeks = math.ceil(difference.days/7)

    for i in range(numWeeks):
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
    firstDate = list(testingDF["Date"].values)[0] #Set this to be the first day of the testingDF
    lastDate = list(dataDF["Date"].values)[-1]
    boughtAtStart = False
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

def bestThresholdInRange(startRange, endRange, step, balance, testingIntervalDF, dataDF, weekDict):
    profitList = []
    threshList = []

    for thresh in np.arange(startRange, endRange, step): #step
        portfolio = Portfolio(balance, _configKeys.YVALUETICKER)
        algorithm_Threshold(portfolio, testingIntervalDF, dataDF, weekDict, thresh, True)
        threshList.append(thresh)
        profitList.append(portfolio.getTotalProfit())
    bestThresh = threshList[profitList.index(max(profitList))]
    bestThresh = round(bestThresh, 3)
    print("Threshold: " + str(bestThresh) + " Profit for best threshold: " + str(max(profitList)))
    return bestThresh

def updateDayTrades(dayTradeList):
    pop = False
    for i in range(len(dayTradeList)):
        if dayTradeList[i] == 0:
            for j in range(len(dayTradeList) - 1):
                dayTradeList[j] = dayTradeList[j+1]
                pop = True
        else:
            dayTradeList[i] -= 1
    if pop == True:
        dayTradeList.pop()

def addDayTrade(dayTradeList):
    dayTradeList.append(5)

def algorithm_Threshold(portfolio, testingDF, dataDF, weekDict, threshold, isTest):
    ticker = portfolio.getTickerName()
    window_length = _configKeys.WINDOW_LENGTH

    firstDate = list(testingDF["Date"].values)[0] #Set this to be the first day of the testingDF
    lastDate = list(dataDF["Date"].values)[-1]

    #list of features we want to track
    featureList = [_configKeys.YVALUETICKER + "_Low_average_Predicted", _configKeys.YVALUETICKER +"_High_average_Predicted"] # make it uniform -- we want gold ticker to be capitalized
    #for each week in our testingDF, get desired predictedFEATURE(s) from the list

    dayTradeList = []
    for week in list(testingDF["Date"].values):
        if threshold != -1:
            #Get the index of the current week within the testing dataframe
            weekRowIndex = testingDF.index[testingDF['Date'] == week][0]

            #Find the low_average_predicted and high_max_predicted for that week
            lowMinP = testingDF.at[weekRowIndex, featureList[0]]
            highMaxP = testingDF.at[weekRowIndex, featureList[1]]
            if (isTest == False):
                print(str(week))
                print ("Threshold for current week: " + str(threshold))
            #Get the dataframe holding the weekly values for this week
            weekDF = weekDict[week]

            #For each daily value in the weekDF
            for day in list(weekDF["Date"].values):
                updateDayTrades(dayTradeList)

                #run a list on the last 5 valid dates? -- then

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
                if (len(dayTradeList) < 3 and isLastDay == False):
                    #Open threshold checks
                    #print ("Open: " + str(nearOpen))
                    #print (str(lowMinP))
                    #print (str(threshold))
                    if (nearOpen < lowMinP + threshold):
                        portfolio.buyMax(nearOpen, day, "Open")
                        transMadeToday = True
                    elif (nearOpen > highMaxP - threshold):
                        portfolio.sellMax(nearOpen, day, "Open")
                        transMadeToday = True

                    #Close thresholds checks
                    if (transMadeToday):
                        if (nearClose < lowMinP + threshold):
                            if transMadeToday:
                                addDayTrade(dayTradeList)
                            portfolio.buyMax(nearClose, day, "Close")
                        elif (nearClose > highMaxP - threshold):
                            portfolio.sellMax(nearClose, day, "Close")
                            if transMadeToday:
                                addDayTrade(dayTradeList)
                    #if we havent trading earlier in the day and we cant day trade anymore, we run algotrader
                    else:
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
                testingIntervalDF = pd.DataFrame(columns = testingDF.columns)
                if weekRowIndex >= window_length:
                    iterDate = inputWeek - timedelta(days=window_length*7) # using this to iter through currentDate - window_length to currentDate

                    for i in range(window_length):
                        weekRow = testingDF.loc[testingDF['Date'] == datetime.datetime.strftime(iterDate, '%Y-%m-%d')]
                        testingIntervalDF = testingIntervalDF.append(weekRow)
                        iterDate = iterDate + timedelta(days=7)
                    '''
                    if weekRowIndex == window_length:
                        print("Starting big range thresh calcl")
                        threshold = bestThresholdInRange(0, nearClose*0.5, nearClose/1000, portfolio.getTotalProfit(), testingIntervalDF, dataDF, weekDict)
                    if weekRowIndex > window_length:
                        print("before new thresh")
                        # everytime, there should be around 500 steps (we add .001 to prevent step = 0 case when threshold = 0)
                        threshold = bestThresholdInRange(threshold*0.15, threshold*2.75+(nearClose*0.08), (threshold*2.75 - threshold*0.15) / 250 + 0.001, portfolio.getTotalProfit(), testingIntervalDF, dataDF, weekDict)
                        print("after new thresh")
                    '''
                    #300 steps every time
                    threshold = bestThresholdInRange(0, nearClose*0.5, nearClose*0.5/300, portfolio.getTotalProfit(), testingIntervalDF, dataDF, weekDict)

main()
