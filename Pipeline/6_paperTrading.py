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


    #findThreshByEstimate(testingDF, dataDF, weeksDict, "date", 1, .2, 1)

    #Algorithms being run on portfolios
    runThresh(thresholdPortfolio, testingDF, dataDF, weeksDict, startBalance)
    runControl(controlPortfolio, testingDF, dataDF, weeksDict)


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


    #loop through all csvs in lasso that have ticker in the filename,
    #keep a list of tuples with filename, percent error in features we care about,
    #reorder the list with minimum to maximum ordering by percent Error
    #choose the filename at the 0th index

#Runs the control algorithm and prints the profit achieved
def runControl(controlPortfolio, testingDF, dataDF, weeksDict):
    algorithm_Control(controlPortfolio, testingDF, dataDF, weeksDict)
    print ("Control Profit: " + str(controlPortfolio.getTotalProfit()))

#Runs the threshold algorithm and prints the profit achieved
def runThresh(thresholdPortfolio, testingDF, dataDF, weeksDict, startBalance):
    #Threshold Testing:
    day = list(dataDF["Date"].values)[-1]
    dayRowIndex = dataDF.index[dataDF['Date'] == day][0]
    price = dataDF.at[dayRowIndex, "Close"]
    algorithm_ApproachThreshold(thresholdPortfolio, testingDF, dataDF, weeksDict, 0, 0, False)
    print("Threshold Algorithm Profit: " + str(thresholdPortfolio.getTotalProfit()))

#Takes in day dataDF, will return a dictionary with the weekly dates as keys and a DataFrame of daily values during that week
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

#Control Algo: Buy first day, sell last day
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

#Finds best threshold through simulating thresholdAlgorithm with an estimated threshold and moving towards the most succesful threshold in jumps
def findThreshByEstimate(testingDF, dataDF, weekDict, date, lowBound, upBound): #do start and end
    window_length = _configKeys.WINDOW_LENGTH
    firstDateString = list(testingDF["Date"].values)[0] #Set this to be the first day of the testingDF
    firstDate = datetime.datetime.strptime(firstDateString, '%Y-%m-%d')
    difference = date.date() - firstDate.date() #difference
    weeksElapsed = math.ceil(difference.days/7) # weeks elapsed is difference in days

    testingIntervalDF = pd.DataFrame(columns = testingDF.columns)

    if weeksElapsed >= window_length:
        iterDate = date - timedelta(days=window_length*7) # using this to iter through currentDate - window_length to currentDate

        for i in range(window_length):
            weekRow = testingDF.loc[testingDF['Date'] == datetime.datetime.strftime(iterDate, '%Y-%m-%d')]
            testingIntervalDF = testingIntervalDF.append(weekRow)
            iterDate = iterDate + timedelta(days=7)

        #if weeksElapsed == window_length:
            #print ("UpBound: " + str(upBound))
            #return bestThresholdInRange(0, upBound, 0.01, currentBalance, testingIntervalDF, dataDF, weekDict)

        midPoint = round((upBound + lowBound)*.5, 5) #upBound will start as the close price of the past week
        qPoint = round(midPoint - (upBound-lowBound)*.25, 5)
        threeQPoint = round(midPoint + (upBound-lowBound)*.25, 5)


        midPortfolio = Portfolio(1000, _configKeys.YVALUETICKER)
        quarterPortfolio = Portfolio(1000, _configKeys.YVALUETICKER) #Maybe use the actual balance that our main portfolio has instead of 1000
        threeQuarterPortfolio = Portfolio(1000, _configKeys.YVALUETICKER)
        lowPortfolio = Portfolio(1000, _configKeys.YVALUETICKER)
        upPortfolio = Portfolio(1000, _configKeys.YVALUETICKER)

        #print()
        #print ("midPoint: " + str(midPoint))
        #print ("qPoint: " + str(qPoint))
        #print ("threeQPoint: " + str(threeQPoint))
        #print ("lowBound: " + str(lowBound))
        #print ("upBound: " + str(upBound))
        #print()

        algorithm_ApproachThreshold(midPortfolio, testingIntervalDF, dataDF, weekDict, midPoint, True)
        algorithm_ApproachThreshold(quarterPortfolio, testingIntervalDF, dataDF, weekDict, qPoint, True)
        algorithm_ApproachThreshold(threeQuarterPortfolio, testingIntervalDF, dataDF, weekDict, threeQPoint, True)
        algorithm_ApproachThreshold(lowPortfolio, testingIntervalDF, dataDF, weekDict, lowBound, True)
        algorithm_ApproachThreshold(upPortfolio, testingIntervalDF, dataDF, weekDict, upBound, True)

        midProfit = midPortfolio.getTotalProfit()
        qProfit = quarterPortfolio.getTotalProfit()
        threeQProfit = threeQuarterPortfolio.getTotalProfit()
        lowProfit = lowPortfolio.getTotalProfit()
        upProfit = upPortfolio.getTotalProfit()



        #print()
        #print("MidProfit: " + str(midProfit))
        #print("QuarterProfit: " + str(qProfit))
        #print("ThreeQuarterProfit: " + str(threeQProfit))
        #print("LowProfit: " + str(lowProfit))
        #print("UpProfit: " + str(upProfit))
        #print()
        #print()

        portfolioProfits = [(midPoint, midProfit), (qPoint, qProfit), (threeQPoint, threeQProfit), (upBound, upProfit), (lowBound, lowProfit)]
        portfolioProfits.sort(key=lambda tup: tup[1], reverse=True)
        #print("Portfolio List: " + str(portfolioProfits))
        #print()
        newBestThreshold = portfolioProfits[0][0]
        newBestProfit = portfolioProfits[0][1]

        #if newBestProfit < bestProfitSoFar:
        #    print("Threshold: " + str(bestThresholdSoFar) + " Profit for best est threshold: " + str(bestProfitSoFar))
        #    return bestThresholdSoFar

        if upBound - lowBound <= .01:
            thr = bestThresholdInRange(lowBound, upBound, .001, testingIntervalDF, dataDF, weekDict) #don't we want 0.01 here??
            #thr = portfolioProfits[0][0]
            print("Thresh from Estimate: " + str(thr))
            return thr

        if newBestThreshold == lowBound:
            return findThreshByEstimate(testingDF, dataDF, weekDict, date, lowBound, qPoint)
        elif newBestThreshold == qPoint:
            return findThreshByEstimate(testingDF, dataDF, weekDict, date, lowBound, midPoint)
        elif newBestThreshold == midPoint:
            return findThreshByEstimate(testingDF, dataDF, weekDict, date, qPoint, threeQPoint)
        elif newBestThreshold == threeQPoint:
            return findThreshByEstimate(testingDF, dataDF, weekDict, date, midPoint, upBound)
        elif newBestThreshold == upBound:
            return findThreshByEstimate(testingDF, dataDF, weekDict, date, threeQPoint, upBound)

    else:
        #print ("Threshold before window length: " + str(threshold))
        return 0

def bestThresholdInRange(startRange, endRange, step, testingIntervalDF, dataDF, weekDict):
    profitList = []
    threshList = []
#    print ("Start: " + str(startRange))
#    print ("End: " + str(endRange))

    for thresh in np.arange(startRange, endRange, step): #step
        portfolio = Portfolio(1000, _configKeys.YVALUETICKER)
        algorithm_ApproachThreshold(portfolio, testingIntervalDF, dataDF, weekDict, thresh, True)
        threshList.append(thresh)
        profitList.append(portfolio.getTotalProfit())
    bestThresh = threshList[profitList.index(max(profitList))]
    bestThresh = round(bestThresh, 3)
    print("Threshold: " + str(bestThresh) + " Profit for best threshold: " + str(max(profitList)))
    return bestThresh

#Threshold Algo: Buys when stock price goes within certain boundaries of predictions
def algorithm_ApproachThreshold(portfolio, testingDF, dataDF, weekDict, botThreshold, topThreshold, isTest):
    ticker = portfolio.getTickerName()
    endWeekPrice = 0 # initializing to keep track so we can feed this to best threshold algorithm to test thresholds appropriately

    firstDate = list(testingDF["Date"].values)[0] #Set this to be the first day of the testingDF
    lastDate = list(dataDF["Date"].values)[-1]
    #list of features we want to track
    predFeatureList = [_configKeys.YVALUETICKER + "_Low_average_Predicted", _configKeys.YVALUETICKER +"_High_average_Predicted"] # make it uniform -- we want gold ticker to be capitalized
    actualFeatureList = [_configKeys.YVALUETICKER + "_Low_average_Actual", _configKeys.YVALUETICKER +"_High_average_Actual"]
    #for each week in our testingDF, get desired predictedFEATURE(s) from the list

    allTimeLowActualAverages = []
    allTimeLowPredictedAverages = []

    pastWindowLowActualAverages = []
    pastWindowLowPredictedAverages = []

    allTimeHighActualAverages = []
    allTimeHighPredictedAverages = []

    pastWindowHighActualAverages = []
    pastWindowHighPredictedAverages = []

    for week in list(testingDF["Date"].values):
            #if threshold != -1:
        #Get the index of the current week within the testing dataframe
        weekRowIndex = testingDF.index[testingDF['Date'] == week][0]

        #Find the low_average_predicted and high_max_predicted for that week
        lowAvgP = testingDF.at[weekRowIndex, predFeatureList[0]]
        highAvgP = testingDF.at[weekRowIndex, predFeatureList[1]]
        if (isTest == False):
            print(str(week))
            print ("Top Threshold for current week: " + str(topThreshold) + " Bot Threshold for current week: " + str(botThreshold))
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
                #print ("Open: " + str(nearOpen))
                #print (str(lowAvgP))
                #print (str(threshold))
                if (nearOpen < lowAvgP + botThreshold and not(nearOpen > highAvgP - topThreshold)):
                    portfolio.buyMax(nearOpen, day, "Open")
                    transMadeToday = True
                elif (nearOpen > highAvgP - topThreshold and not(nearOpen < lowAvgP + botThreshold)):
                    portfolio.sellMax(nearOpen, day, "Open")
                    transMadeToday = True

                #Close thresholds checks
                #if we have not max day traded, then we do this
                if daytradeCount < 3:
                    if (nearClose < lowAvgP + botThreshold):
                        if transMadeToday:
                            daytradeCount += 1
                        portfolio.buyMax(nearClose, day, "Close")
                    elif (nearClose > highAvgP - topThreshold):
                        portfolio.sellMax(nearClose, day, "Close")
                        if transMadeToday:
                            daytradeCount += 1
                #if we havent trading earlier in the day and we cant day trade anymore, we run algotrader
                elif (transMadeToday == False):

                    if (nearClose < lowAvgP + botThreshold and not(nearClose > highAvgP - topThreshold)):
                        portfolio.buyMax(nearClose, day, "Close")
                    elif (nearClose > highAvgP - topThreshold and not(nearClose < lowAvgP + botThreshold)):
                        portfolio.sellMax(nearClose, day, "Close")
                #If we are in the last day of the trading week, sell all we have at open or close
            if (isLastDay):
                    if (nearOpen > highAvgP - topThreshold):
                        portfolio.sellMax(nearOpen, day, "Open")
                    else:
                        portfolio.sellMax(nearClose, day, "Close")

        inputWeek = datetime.datetime.strptime(week, "%Y-%m-%d")

        #Keeping track of our low actual and predicted
    #    allTimeLowActualAverages.append(testingDF.at[weekRowIndex, actualFeatureList[0]])
    #    allTimeLowPredictedAverages.append(lowAvgP)

        if len(pastWindowLowActualAverages) == 13: #If we are at window length
            #Remove the oldest window averages
            pastWindowLowActualAverages.pop(0)
            pastWindowLowPredictedAverages.pop(0)
            #Add the newest window averages
            pastWindowLowActualAverages = [testingDF.at[weekRowIndex, actualFeatureList[0]]]
            pastWindowLowPredictedAverages = [lowAvgP]
        else: #For when we haven't gone through thirteen weeks yet
            pastWindowLowActualAverages = [testingDF.at[weekRowIndex, actualFeatureList[0]]]
            pastWindowLowPredictedAverages = [lowAvgP]

        #Keeping track of our high actual and predicted
    #    allTimeHighActualAverages = [testingDF.at[weekRowIndex, actualFeatureList[1]]]
    #    allTimeHighPredictedAverages.append(highAvgP)

        if len(pastWindowLowActualAverages) == 13: #If we are at window length
            #Remove the oldest window averages
            pastWindowHighActualAverages.pop(0)
            pastWindowHighPredictedAverages.pop(0)
            #Add the newest window averages
            pastWindowHighActualAverages = [testingDF.at[weekRowIndex, actualFeatureList[1]]]
            pastWindowHighPredictedAverages = [highAvgP]
        else: #For when we haven't gone through thirteen weeks yet
            pastWindowHighActualAverages = [testingDF.at[weekRowIndex, actualFeatureList[1]]]
            pastWindowHighPredictedAverages = [highAvgP]


        if isTest == False:
            #threshold = findThreshByEstimate(testingDF, dataDF, weekDict, inputWeek, 0, nearClose)

        #    botThreshold = calculateMeanError(allTimeLowActualAverages, allTimeLowPredictedAverages) #errorInLowAvg for all time as threshold
        #    topThreshold = calculateMeanError(allTimeHighActualAverages, allTimeHighPredictedAverages) #errorInHighAvg for all time as threshold

            #This is the best consistent profit for Gold:
            botThreshold = calculateMeanError(pastWindowLowActualAverages, pastWindowLowPredictedAverages) #errorInLowAvg for window as threshold
            topThreshold = calculateMeanError(pastWindowHighActualAverages, pastWindowHighPredictedAverages) #errorInHighAvg for window as threshold

        '''
        For tomorrow:
        Do this for [errorInLowAvg for threshold, errorInHighAvg for threshold, errorInLowAvg for selling threshold and errorInHighAvg for buying threshold]
        use calculateMeanError on all time data
        use calulateMeanError on past 13 weeks
        maybe make functions
        '''
            #threshold = .011

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
1. Mean Error as decider of how many shares to buy
2. Swing Trading: Using bullish and bearish logic to maximize earnings during deviations from the "main trend line": https://www.ally.com/do-it-right/investing/swing-trading-strategy-guide/
'''

main()
