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
        self.totalPrice = self.shareCost * self.numShares

    def __str__(self):
         return "Ticker: " + str(self.ticker) + ",  Share Cost: " + str(self.shareCost) + ",  Num Shares: " + str(self.numShares) + ",  Date: " + str(self.date) + ",  Transaction type: " + str(self.transactionType) + ",  Time: " + str(self.time)

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

def main():
    #for friday: have control algorithm run, and have algorithm #1 run, and output it somehow with a csv
    startBalance = 1000
    window_length = _configKeys.WINDOW_LENGTH


    thresholdPortfolio = Portfolio(startBalance, _configKeys.YVALUETICKER)
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

    #print(meanErrorList)

    #algorithms being run on portfolios
    algorithm_Control(controlPortfolio, testingDF, dataDF, weeksDict)

    #numShares = 100
    profitList = []
    '''
    IDEA FOR THRESHOLDS: By looking at the mean change in low avg / high max in a window lenght, we may be able to optimize our profits by dynamically changing the threshold according to the window length

    '''
    '''
    #Threshold Testing:
    for i in range(48, 68):
        numShares = 10
        shareList = []
        for j in range(0,9):
            thresholdPortfolio = Portfolio(startBalance, _configKeys.YVALUETICKER)
            algorithm_ApproachThreshold(thresholdPortfolio, testingDF, dataDF, weeksDict, i / 1000, numShares)
            shareList.append(thresholdPortfolio.getTotalProfit())
            numShares += 10

        profitList.append("Threshold: " + str(i/1000) + ": Best Num Shares: " + str((shareList.index(max(shareList))+1)*10) + ": Max Profit for threshold with best num shares: " + str(max(shareList)))

    print("Threshold Algorithm Results:")
    print()
    for prof in profitList:
        print (prof)
        print()
    '''
    numSharesLimit = 10
    shareListLimit = []
    for j in range(0,10):
        limitPortfolio = Portfolio(startBalance, _configKeys.YVALUETICKER)
        algorithm_Limits(limitPortfolio, testingDF, dataDF, weeksDict, numSharesLimit)
        shareListLimit.append(limitPortfolio.getTotalProfit())
        numSharesLimit += 10

    print("Limit Algorithm Results:")
    print()
    for i in range(0,10):
        print ("Num shares"  + str((i+1)*10) + ": Max Profit with best num shares: " + str(shareListLimit[i]))
        print()

    print ("Control Profit: " + str(controlPortfolio.getTotalProfit()))


#Take in day dataDF
#Will return a dictionary with the weekly dates as keys and a DataFrame of daily values during that week
def daysInWeekDict(dataDF):
    #TODO: make this return what we want (a DataFrame with all trading days for a specific week)
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
        for i in range(5):
            currentDate = date + datetime.timedelta(days=i)
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



def algorithm_MeanError(portfolio, testingDF, dataDF, weekDict):
    #only use error for a specific window
    #nextPrice - currPrice - meanError is positive -> buy
    ticker = portfolio.getTickerName()
    for week in list(testingDF["Date"].values):
        #grabbing daily values for this week (we have the daily data in a DataFrame with 5 values)
        weekDF = weekDict[week]
        weekIndex = dataDF.index[dataDF['Date'] == day][0]

        openPred = testingDF.at(weekIndex, _configKeys.YVALUETICKER + "_Open_max_Predicted")
        closePred = testingDF.at(weekIndex, _configKeys.YVALUETICKER + "_Close_average_Predicted")
        volitilityPred = testingDF.at(weekIndex, _configKeys.YVALUETICKER + "_Open_volatility_Predicted")
        '''
        openMeanError = calculateMeanError
        closeMeanError =
        volitilityMeanError =
        #these are values we want to compare each daily ACTUAL value (Formatting -- maybe just make every word start with a capital Letter -- bring up in meeting)
        for day in list(weekDF["Date"].values):
        '''

        #for day in list(daysDF["Date"].values):


#idea- buy, and only sell if it goes above a predicted max


'''
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
            low = dataDF.at[dayRowIndex, "Low"]
            high = dataDF.at[dayRowIndex, "High"]

                #If statement
                #If the actual low is within our predicted low plus some threshold, we buy the next trading day
            if (list(weekDF["Date"].values).index(day) != len(list(weekDF["Date"].values))-1):
                tomorrow = dataDF.at[dayRowIndex + 1, "Date"]
                tomorrowOpen = dataDF.at[dayRowIndex + 1, "Open"]
                if (low <= lowAvgP): #Num shares bought is 20% of total available shares
                    transaction = Transaction("GOLD", tomorrowOpen, numShares, tomorrow, "Buy", "Open")
                    portfolio.buyStock(transaction)
                    '''
                    print(str(transaction))
                    print ("Low average predicted" + str(lowAvgP))
                    print ("High max predicted" + str(highMaxP))
                    '''
                #If the actual high is within our predicted high minus some threshold, we sell the next trading day
                if (high >= highMaxP):
                    transaction = Transaction("GOLD", tomorrowOpen, numShares, tomorrow, "Sell", "Open")
                    portfolio.sellStock(transaction)
                    '''
                    print(str(transaction))
                    print ("Low average predicted" + str(lowAvgP))
                    print ("High max predicted" + str(highMaxP))
                    '''
            #Sell on last day always to liquidate assets
            if day == lastDate:
                dayRowIndex = dataDF.index[dataDF['Date'] == day][0]
                price = dataDF.at[dayRowIndex, "Close"]
                transaction = Transaction("GOLD", price, 700, day, "Sell", "Close")
                #ticker, shareCost, numShares, date, transactionType, time
                portfolio.sellStock(transaction)

def algorithm_ApproachThreshold(portfolio, testingDF, dataDF, weekDict, threshold, numShares):
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
            low = dataDF.at[dayRowIndex, "Low"]
            high = dataDF.at[dayRowIndex, "High"]

                #If statement
                #If the actual low is within our predicted low plus some threshold, we buy the next trading day
            if (list(weekDF["Date"].values).index(day) != len(list(weekDF["Date"].values))-1):
                tomorrow = dataDF.at[dayRowIndex + 1, "Date"]
                tomorrowOpen = dataDF.at[dayRowIndex + 1, "Open"]
                if (low < lowAvgP + threshold): #Num shares bought is 20% of total available shares
                    transaction = Transaction("GOLD", tomorrowOpen, numShares, tomorrow, "Buy", "Open")
                    portfolio.buyStock(transaction)
                    '''
                    print(str(transaction))
                    print ("Low average predicted" + str(lowAvgP))
                    print ("High max predicted" + str(highMaxP))
                    '''
                #If the actual high is within our predicted high minus some threshold, we sell the next trading day
                if (high > highMaxP - threshold):
                    transaction = Transaction("GOLD", tomorrowOpen, numShares, tomorrow, "Sell", "Open")
                    portfolio.sellStock(transaction)
                    '''
                    print(str(transaction))
                    print ("Low average predicted" + str(lowAvgP))
                    print ("High max predicted" + str(highMaxP))
                    '''
            #Sell on last day always to liquidate assets
            if day == lastDate:
                dayRowIndex = dataDF.index[dataDF['Date'] == day][0]
                price = dataDF.at[dayRowIndex, "Close"]
                transaction = Transaction("GOLD", price, 700, day, "Sell", "Close")
                #ticker, shareCost, numShares, date, transactionType, time
                portfolio.sellStock(transaction)



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

'''
def GetWeekDictionary(assetDF, include_volume):


    #This piece of code breaks up the daily csv into weeks

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
