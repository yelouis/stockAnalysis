#This file will keep track of all the moving variables and we can slowly add to that file
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

def main():
    '''
    This is the section of code where you can specify what stocks/funds/bonds/etc you'd like to get and how to bin them by week
    Data csv columns:

    Stock: Date, Open, High, Low, Close, Volume, Currency
    Fund: Date, Open, High, Low, Close, Currency
    Bonds: Date, Open, High, Low, Close
    Commodity: Date, Open, High, Low, Close, Volume, Currency

    Note: Thankfully, we likely won't need to use the Currency column (since it is always USD).
    This allows us to use the same CollapseDictionaryToWeeks() function for all asset classes
    '''

    successfulBins = {"Symbol" : [],
                     "Type" : []}

    # We will use the 1successfulPulls.csv to tell us what type of asset is associated with each name/ticker
    reference_df = pd.read_csv("2successfulDayBins.csv", low_memory=False)
    window_length = _configKeys.WINDOW_LENGTH
    reference_days_list = list(pd.read_csv(os.path.join(Path("2Day_Binned"), "BA"+".csv"), low_memory=False)["Date"])
    #index is an asset row
    for index in reference_df.index:
        name = reference_df["Symbol"][index]
        print(name)
        asset_class = reference_df["Type"][index]

        '''
        asset_class_has_volume = False
        if asset_class == "Stock" or asset_class == "Commodity":
            asset_class_has_volume = True
        '''
        assetDF = pd.read_csv(os.path.join(Path("2Day_Binned"), name+".csv"), low_memory=False)
        if list(assetDF["Date"]) != reference_days_list:
            print("not enough days")
            continue

        for i in ["Open", "Close", "High", "Low", "Volume", "Currency", "Exchange"]:
            if i in assetDF.columns:
                del assetDF[i]

        assetDF.to_csv(os.path.join(Path(_configKeys.STANDARDIZED_FOLDER), name+".csv"), index=False)

        #Update the successful bins dataframe
        successfulBins["Symbol"].append(name)
        successfulBins["Type"].append(asset_class)

    successfulBins = pd.DataFrame(successfulBins, columns = ["Symbol", "Type"])
    #Creating a sucessful file that includes asset names/tickers
    successfulBins.to_csv('3successfulStandardizedBins.csv', index=False)
    #Now we throw this dataframe into a csv file and add it to a 2successfulWeekBins.csv


main()
