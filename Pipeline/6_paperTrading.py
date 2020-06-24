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



    def nonRecursionBuyMax(self, price, date, time):
        startingShares = self.numSharesOwned
        startingBalance = self.balance
        maxSharesCanBuy = math.floor(self.balance / price)
        if (maxSharesCanBuy != 0):
            self.balance -= maxSharesCanBuy * price
            self.numSharesOwned += maxSharesCanBuy
            self.transactions.append(((price, date, "Buy", time), ((", Shares before transaction: " + str(startingShares) + " After transaction: " + str(self.numSharesOwned), "Balance before transaction: " + str(startingBalance) + " After transaction: " + str(self.balance)))))



    def buyMax(self, transaction): #will take any transaction and make it buy max shares
        if self.validBuyTransaction(transaction) == False:
            self.buyStock(transaction)
        else:
            transaction.oneMoreShare()
            self.buyMax(transaction)
            '''
            self.numSharesOwned += transaction.getNumShares()
            self.balance -= transaction.getTotalPrice()
            self.transactions.append((transaction, (self.numSharesOwned, self.balance)))
            '''

    def sellMax(self, price, date, time): #will take any transaction and make it sell shares
        startingShares = self.numSharesOwned
        startingBalance = self.balance
        if (self.numSharesOwned != 0):
            self.balance += self.numSharesOwned * price
            self.numSharesOwned = 0
            self.transactions.append((("Price: " + str(price), "Date: " + str(date), "Sell", "Time: " + str(time)), ((", Shares before transaction: " + str(startingShares) + " After transaction: " + str(self.numSharesOwned), "Balance before transaction: " + str(startingBalance) + " After transaction: " + str(self.balance)))))

    #displayTransactions will print out all the transactions in the Portfolio within the timeframe specified
    def displayTransactions(self, startDate, endDate):
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
            (transaction, data) = trans
            print(transaction, data)
            print()


    #validBuyTransaction will return whether we can complete a buy transaction
    def validBuyTransaction(self, transaction):
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
        self.date = date #
        self.transactionType = transactionType # transactionType will be a string "Buy" or "Sell"
        self.time = time # time will be either "open" or "close" to determine if we are buying/selling at open or close
        self.totalPrice = self.shareCost * self.numShares

    def __str__(self):
         return "Ticker: " + str(self.ticker) + ",  Share Cost: " + str(self.shareCost) + ",  Num Shares: " + str(self.numShares) + ",  Date: " + str(self.date) + ",  Transaction type: " + str(self.transactionType) + ",  Time: " + str(self.time)

    #getDate will return the date of the transaction
    def getDate(self):
        return self.date

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
        self.totalPrice = self.shareCost * self.numShares

    def oneMoreShare(self):
        self.numShares += 1
        self.totalPrice = self.shareCost * self.numShares

def main():
    #for friday: have control algorithm run, and have algorithm #1 run, and output it somehow with a csv
    startBalance = 1000
    window_length = _configKeys.WINDOW_LENGTH
    thresholdPortfolio = Portfolio(startBalance, _configKeys.YVALUETICKER)
    controlPortfolio = Portfolio(startBalance, _configKeys.YVALUETICKER)

    testingDF = pd.read_csv(os.path.join(Path(_configKeys.TESTING_RESULTS_FOLDER), "GOLD0.3_alpha13_beta_test_results.csv"))
    dataDF = pd.read_csv(os.path.join(Path(_configKeys.DATA_FOLDER), "GOLD.csv"))
    weeksDict = daysInWeekDict(dataDF)

    #algorithms being run on portfolios
    runControl(controlPortfolio, testingDF, dataDF, weeksDict)
    runThresh(thresholdPortfolio, testingDF, dataDF, weeksDict, startBalance)
    #runLimit(testingDF, dataDF, weeksDict, startBalance)



def runControl(controlPortfolio, testingDF, dataDF, weeksDict):
    algorithm_Control(controlPortfolio, testingDF, dataDF, weeksDict)
    print ("Control Profit: " + str(controlPortfolio.getTotalProfit()))

def runThresh(thresholdPortfolio, testingDF, dataDF, weeksDict, startBalance):
    #Threshold Testing:
    threshold = 0.051
    algorithm_ApproachThreshold(thresholdPortfolio, testingDF, dataDF, weeksDict, threshold, False)
    print("Threshold Algorithm Profit: " + str(thresholdPortfolio.getTotalProfit()))


    '''
    CAN IGNORE- we are keeping this code incase we have any reason to manually test thresholds/numshares by incrementing in the future
    profitList = []
    for i in range(48, 68):
        numShares = 50
        shareList = []
        for j in range(0,11):
            thresholdPortfolio = Portfolio(startBalance, _configKeys.YVALUETICKER)
            algorithm_ApproachThreshold(thresholdPortfolio, testingDF, dataDF, weeksDict, i / 1000, numShares)
            shareList.append(thresholdPortfolio.getTotalProfit())
            #print ("Threshold: " + str(i/1000) + "Shares: " + str(numShares) + ", Profit is: " + str(thresholdPortfolio.getTotalProfit()))
            numShares += 50
        print()
        profitList.append("Threshold: " + str(i/1000) + ": Best Num Shares: " + str(50 + shareList.index(max(shareList))*50) + ": Max Profit for threshold with best num shares: " + str(max(shareList)))
        print("Transactions for threshold: " + str(i))
        thresholdPortfolio.displayAllTransactions()
        print()
    print()
    print("Threshold Algorithm Results:")
    print()
    for prof in profitList:
        print (prof)
        print()
    '''

def runLimit(testingDF, dataDF, weeksDict, startBalance):
    numSharesLimit = 50
    shareListLimit = []
    for j in range(0,11):
        limitPortfolio = Portfolio(startBalance, _configKeys.YVALUETICKER)
        algorithm_Limits(limitPortfolio, testingDF, dataDF, weeksDict, numSharesLimit)
        shareListLimit.append(limitPortfolio.getTotalProfit())
        numSharesLimit += 50

    print("Limit Algorithm Results:")
    print()
    for i in range(0,10):
        print ("Num shares"  + str(50*i) + ": Max Profit with best num shares: " + str(shareListLimit[i]))
        print()

#Take in day dataDF
#Will return a dictionary with the weekly dates as keys and a DataFrame of daily values during that week
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
        for j in range(5):
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

    for week in list(testingDF["Date"].values):
        weekDF = weekDict[week]
        for day in list(weekDF["Date"].values):
            if day == firstDate:
                dayRowIndex = dataDF.index[dataDF['Date'] == day][0]
                price = dataDF.at[dayRowIndex, "Open"]
                transaction = Transaction("GOLD", price, 700, day, "Buy", "Open")
                #ticker, shareCost, numShares, date, transactionType, time
                portfolio.buyStock(transaction)
            if day == lastDate:
                dayRowIndex = dataDF.index[dataDF['Date'] == day][0]
                price = dataDF.at[dayRowIndex, "Close"]
                transaction = Transaction("GOLD", price, 700, day, "Sell", "Close")
                #ticker, shareCost, numShares, date, transactionType, time
                portfolio.sellStock(transaction)


'''
CAN IGNORE-- we may want to still consider a changing the num shares we buy
Smarter Threshold algorithm:
    1. More speciic buying / selling (E.g. if a stock is within the predicted high minus the threshold, sell x# of stock, if it's within the predicted high + threshold, sell x#*2 of stock)
    2. Looking at past week actuals versus current week predicted
        a. (E.g Actual high of first week of trading was higher than current week of trading high, therefore we do no selling no matter what)


counter:
    buy: 30 shares @ $15
    sell: 15 shares
    If shares fall below 7% we sell, and go back to predictions to decide if we invest
    buy: 30 shares @ $12
    now stonk goes down to 8
'''

#our algo trader without thresholds
def algorithm_Limits(portfolio, testingDF, dataDF, weekDict, numShares):
    ticker = portfolio.getTickerName()

    firstDate = list(testingDF["Date"].values)[0] #Set this to be the first day of the testingDF
    lastDate = list(dataDF["Date"].values)[-1]

    #Idea: ex- If the actual average High of first week of trading is greater than the predicted average high of second week, we do no buying at all no matter what

    #list of features we want to track
    #featureList = [ticker+"_Open_max_Predicted",ticker+"_High_volatility_Predicted", ticker+"_High_max_Predicted"]
    featureList = ["Gold_Low_average_Predicted", "Gold_High_max_Predicted"]
    lastDate = list(dataDF["Date"].values)[-1]

    #for each week in our testingDF, get desired predictedFEATURE(s) from the list
    for week in list(testingDF["Date"].values):
        #Get the index of the current week within the testing dataframe
        weekRowIndex = testingDF.index[testingDF['Date'] == week][0]

        #Find the low_average_predicted and high_max_predicted for that week
        lowAvgP = testingDF.at[weekRowIndex, featureList[0]]
        highMaxP = testingDF.at[weekRowIndex, featureList[1]]


        #Get the dataframe holding the weekly values for this week
        weekDF = weekDict[week]

        for day in list(weekDF["Date"].values): #For each daily value in the weekDF

            #Get the index of the current day within the weekly dataframe
            dayRowIndex = dataDF.index[dataDF['Date'] == day][0]
            #Get the low and high values of the current day AT THE END OF THE DAY (3:59EST) IN ORDER TO MAKE PURCHASE AT OPEN OF NEXT DAY
            #we approximate the 9:30:01AM EST (when we would actually buy/sell) value of the stock with the close price (In practice, we would simply get the stock price at that point and run it through our buy / sell algorithm)
            nearOpen = dataDF.at[dayRowIndex, "Open"]
            #we approximate the 3:59:59PM EST (when we would actually buy/sell) value of the stock with the close price (In practice, we would simply get the stock price at that point and run it through our buy / sell algorithm)
            nearClose = dataDF.at[dayRowIndex, "Close"]

            #lowThreshold = calculateMeanError()
            #highThreshold = calculateMeanError

            if (nearOpen < lowAvgP):
                transaction = Transaction("GOLD", nearOpen, numShares, day, "Buy", "Open")
                portfolio.buyStock(transaction)
            if (nearOpen > highMaxP):
                transaction = Transaction("GOLD", nearOpen, numShares, day, "Sell", "Open")
                portfolio.sellStock(transaction)

            if (nearClose < lowAvgP):
                transaction = Transaction("GOLD", nearClose, numShares, day, "Buy", "Close")
                portfolio.buyStock(transaction)
            if (nearClose > highMaxP):
                transaction = Transaction("GOLD", nearClose, numShares, day, "Sell", "Close")
                portfolio.sellStock(transaction)

            #Sell on last day always to liquidate assets
            if day == lastDate:
                dayRowIndex = dataDF.index[dataDF['Date'] == day][0]
                price = dataDF.at[dayRowIndex, "Close"]
                transaction = Transaction("GOLD", price, 700, day, "Sell", "Close")
                #ticker, shareCost, numShares, date, transactionType, time
                portfolio.sellStock(transaction)

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
    ticker = portfolio.getTickerName()
    endWeekPrice = 0 # initializing to keep track so we can feed this to best threshold algorithm to test thresholds appropriately

    firstDate = list(testingDF["Date"].values)[0] #Set this to be the first day of the testingDF
    lastDate = list(dataDF["Date"].values)[-1]
    #list of features we want to track
    featureList = ["Gold_Low_average_Predicted", "Gold_High_max_Predicted"]
    #for each week in our testingDF, get desired predictedFEATURE(s) from the list
    for week in list(testingDF["Date"].values):
        #Get the index of the current week within the testing dataframe
        weekRowIndex = testingDF.index[testingDF['Date'] == week][0]

        #Find the low_average_predicted and high_max_predicted for that week
        lowAvgP = testingDF.at[weekRowIndex, featureList[0]]
        highMaxP = testingDF.at[weekRowIndex, featureList[1]]


        #Get the dataframe holding the weekly values for this week
        weekDF = weekDict[week]

        daytradeCount = 0
        for day in list(weekDF["Date"].values): #For each daily value in the weekDF
            #Get the index of the current day within the weekly dataframe
            dayRowIndex = dataDF.index[dataDF['Date'] == day][0]

            #we approximate the 9:30:01AM EST (when we would actually buy/sell) value of the stock with the close price (In practice, we would simply get the stock price at that point and run it through our buy / sell algorithm)
            nearOpen = dataDF.at[dayRowIndex, "Open"]
            #we approximate the 3:59:59PM EST (when we would actually buy/sell) value of the stock with the close price (In practice, we would simply get the stock price at that point and run it through our buy / sell algorithm)
            nearClose = dataDF.at[dayRowIndex, "Close"]
            endWeekPrice = nearClose
            #lowThreshold = calculateMeanError()
            #highThreshold = calculateMeanError
            notLast = False
            #convert this to elif logic -- can buy/sell at open/close but we want to avoid day trading
            if day == list(weekDF["Date"].values)[-1]:
                notLast = True
            #Allows for 3 day trade days at the most and will run on every trading day except last (on last day, we only sell)
            if (daytradeCount < 3 and notLast == False):
                #Open threshold checks
                if (nearOpen < lowAvgP + threshold):
                    portfolio.nonRecursionBuyMax(nearOpen, day, "Open")
                    transMadeToday = True
                elif (nearOpen > highMaxP - threshold):
                    portfolio.sellMax(nearOpen, day, "Open")
                    transMadeToday = True

                #Close thresholds checks
                #if we have not max day traded, then we do this
                if daytradeCount < 3:

                    if (nearClose < lowAvgP + threshold):
                        if transMadeToday:
                            daytradeCount += 1
                        portfolio.nonRecursionBuyMax(nearClose, day, "Close")
                    elif (nearClose > highMaxP - threshold):
                        portfolio.sellMax(nearClose, day, "Close")
                        if transMadeToday:
                            daytradeCount += 1
                #if we havent trading earlier in the day and we cant day trade anymore, we run algotrader
                elif (transMadeToday == False):

                    if (nearClose < lowAvgP + threshold):
                        portfolio.nonRecursionBuyMax(nearClose, day, "Close")
                    elif (nearClose > highMaxP - threshold):
                        portfolio.sellMax(nearClose, day, "Close")
            #If we are in the last day of the trading week, sell all we have at open or close
            if (day == list(weekDF["Date"].values)[-1]):
                    if (nearOpen > highMaxP - threshold):
                        #transaction = Transaction("GOLD", nearOpen, numShares, day, "Sell", "Open")
                        #portfolio.sellStock(transaction)
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
