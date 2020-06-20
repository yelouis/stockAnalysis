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

global sucessfulBins
sucessfulBins = {"Symbol" : [],
                 "Type" : []}

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
    # We will use the 1successfulPulls.csv to tell us what type of asset is associated with each name/ticker
    reference_df = pd.read_csv("1successfulPulls.csv", low_memory=False)

    assetDF = pd.read_csv(os.path.join(Path(_configKeys.DATA_FOLDER), _configKeys.YVALUETICKER+".csv"), low_memory=False)
    usable_dates = list(assetDF["Date"])

    for index in reference_df.index:

        name = reference_df["Symbol"][index]
        print(name)
        asset_class = reference_df["Type"][index]

        asset_class_has_volume = False
        if asset_class == "Stock" or asset_class == "Commodity":
            asset_class_has_volume = True

        assetDF = pd.read_csv(os.path.join(Path(_configKeys.DATA_FOLDER), name+".csv"), low_memory=False)

        change_percent = []
        prop_max_volume = []
        include_change_percent = True
        include_volume = True
        for day in assetDF.index:
            if assetDF["Open"][day] == 0:
                include_change_percent = False
            else:
                change_percent.append(100 * (assetDF["Close"][day] - assetDF["Open"][day])/assetDF["Open"][day])
            if asset_class_has_volume:
                if max(list(assetDF["Volume"][:day + 1])) == 0:
                    include_volume = False
                else:
                    prop_max_volume.append(assetDF["Volume"][day] / max(list(assetDF["Volume"][:day + 1])))

        if include_change_percent:
            assetDF[name + "_change%"] = change_percent
        if asset_class_has_volume and include_volume:
            assetDF[name + "_volume_proportion"] = prop_max_volume

        dates_to_drop = [0]
        rows_to_drop = []
        dList = list(assetDF["Date"])
        for row_num in range(len(dList)):
            if dList[row_num] not in usable_dates:
                dates_to_drop.append(dList[row_num])
                rows_to_drop.append(row_num)

        use_stock = True
        for date in usable_dates:
            if date not in dList:
                use_stock = False

        if use_stock == False:
            continue
        else:
            assetDF = assetDF.drop(rows_to_drop)

            assetDF = assetDF.iloc[50:]

            assetDF.to_csv(os.path.join(Path("2Day_Binned"), name+".csv"), index=False)

            #Update the successful bins dataframe
            sucessfulBins["Symbol"].append(name)
            sucessfulBins["Type"].append(asset_class)

    df = pd.DataFrame(sucessfulBins, columns = ["Symbol", "Type"])
    #Creating a sucessful file that includes asset names/tickers
    df.to_csv('2successfulDayBins.csv', index=False)
    #Now we throw this dataframe into a csv file and add it to a 2successfulWeekBins.csv


main()
