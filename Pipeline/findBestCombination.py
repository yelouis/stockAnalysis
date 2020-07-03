#THE PURPOSE OF THIS FILE IS TO FIND THE BEST LINEAR REGRESSION METHOD FOR PREDICTING A STOCK ALONG WITH THAT (STOCK, METHOD)'S BEST ALPHA AND BETA VALUES
import _configKeys
import os
from pathlib import Path
import pandas as pd
import csv


def main():
    makeProfitComparisonCSVS()
    print("Hello World!")

'''
makeProfitComparisonCSVS with look at all of the results in the 6Paper_Results folder and make a csv for every stock showcasing the profit in this format:

                   Elastic Profit   Lasso Profit    Control Profit
beta1 w/ best alpha       x              x                 x
beta2 w/ best alpha       x              x
beta3 w/ best alpha       x              x
beta4 w/ best alpha       x              x

x being the datapoint at the corresponding cell
'''

def makeProfitComparisonCSVS():
    tickerDict = {}
    for filename in os.listdir(os.path.join(Path(_configKeys.PAPER_RESULTS_FOLDER))):
        ticker = filename.rsplit(sep=".")[0][:-1]
        ticker = ticker.split("_con")[0]
        if ticker not in tickerDict:
            tickerDict[ticker] = [filename]
        else:
            tickerDict[ticker].append(filename)

    for ticker in tickerDict:
        abDict = {}
        controlProfit = 0
        for algoFile in tickerDict[ticker]:
            profit = 0
            fileType = ""
            abValue = algoFile.strip(ticker)
            algoProfitDF = pd.read_csv(os.path.join(Path(_configKeys.PAPER_RESULTS_FOLDER), algoFile))
            endBalance = list(algoProfitDF["Balance After"].values)[-1]
            profit = endBalance - 1000
            if "control" in algoFile:
                controlProfit = profit
            elif "lasso" in algoFile:
                fileType = "lasso"
                abValue = abValue.split("_lasso_threshold")[0]
                recordAlphaBeta(abDict, abValue, profit, fileType)
            elif "elastic" in algoFile:
                fileType = "elastic"
                abValue = abValue.split("_elastic_threshold")[0]
                recordAlphaBeta(abDict, abValue, profit, fileType)
        abUnsortedRows = abDict.items()
        abSortedRows = sorted(abUnsortedRows)
        dfList = []
        for i in range(len(abSortedRows)):
            row = abSortedRows[i]
            abSortedRows[i][1].append(controlProfit)
            dfList.append([row[0]] + row[1])
        tickerDF = pd.DataFrame(dfList, columns=["Beta / Alpha", "Elastic Profit", "Lasso Profit", "Control Profit"])
        tickerDF.to_csv(os.path.join(Path(_configKeys.PROFIT_COMPARISONS_FOLDER), ticker+"_profit_comparison.csv"))

def recordAlphaBeta(dictionary, alphabeta, profit, fileType): #dictionary to check in, alphabeta to check for, profit is the profit to keep track of, fileType to say which index to add the profit into
    if alphabeta not in dictionary:
        if fileType == "lasso":
            dictionary[alphabeta] = ["N/A", profit]
        elif fileType == "elastic":
            dictionary[alphabeta] = [profit, "N/A"]
    else:
        if fileType == "lasso":
            dictionary[alphabeta][1] = profit #This case implies we already have a recorded elastic profit structured as dict[abValue] = [elasticProfit, "N/A"]
        elif fileType == "elastic":
            dictionary[alphabeta][0] = profit #This case implies we already have a recorded lasso profit structured as dict[abValue] = ["N/A", lassoProfit]

main()
