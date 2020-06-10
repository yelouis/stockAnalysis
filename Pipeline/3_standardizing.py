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


def find_asset_class(name, reference_df):
    for index in reference_df.index:
        if reference_df["Symbol"][index] == name:
            return reference_df["Type"][index]
    print("could not find " + name + " in the reference csv")
    quit()


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

    for asset in name_element_stat:
        name = reference_df["Symbol"][index]
        asset_class = reference_df["Type"][index]

        asset_class_has_volume = False
        if asset_class == "Stock" or asset_class == "Commodity":
            asset_class_has_volume = True

        assetDF = pd.read_csv(os.path.join(Path(_configKeys.DATA_FOLDER), name+".csv"), low_memory=False)

        asset_dictionary = GetWeekDictionary(assetDF, asset_class_has_volume)

        week_bin_df = CollapseDictionaryToWeeks(asset_dictionary, name, asset_class_has_volume)
        week_bin_df.to_csv(os.path.join(Path(_configKeys.BINNED_FOLDER), name+".csv"), index=False)

        #Update the successful bins dataframe
        sucessfulBins["Symbol"].append(name)
        sucessfulBins["Type"].append(asset_class)

    df = pd.DataFrame(sucessfulBins, columns = ["Symbol", "Type"])
    #Creating a sucessful file that includes asset names/tickers
    df.to_csv('2successfulWeekBins.csv', index=False)
    #Now we throw this dataframe into a csv file and add it to a 2successfulWeekBins.csv


main()
