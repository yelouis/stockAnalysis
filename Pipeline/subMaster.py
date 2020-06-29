# This will be the subMasterfile that runs the python files from 1-5.
import _configKeys
import pandas as pd
#pipelineP1 = __import__('1_stockImporting.py')

global successfulBins
successfulBins = {"Symbol" : [],
                 "Type" : []}


def main():

    _configKeys.STARTPULL = "03/01/2016"
    _configKeys.ENDPULL = "07/01/2018"
    _configKeys.FIRSTINDEX = "2016-01-03"
    _configKeys.LASTINDEX = "2018-01-07"
    _configKeys.DATA_FOLDER = "1Data_2016/"
    _configKeys.BINNED_FOLDER = "2Binned_2016/"
    _configKeys.STANDARDIZED_FOLDER = "3Standardized_Binned_2016/"
    _configKeys.SUCCESSFULSTANDARDIZEDBINS = "3successfulStandardizedBins_2016.csv"
    pipelineP1a = __import__('1_stockImporting.py')
    pipelineP2a = __import__('2_binning.py')
    pipelineP3a = __import__('3_standardizing.py')

    _configKeys.STARTPULL = "14/01/2018"
    _configKeys.ENDPULL = "05/01/2020"
    _configKeys.FIRSTINDEX = "2018-01-14"
    _configKeys.LASTINDEX = "2020-01-05"
    _configKeys.DATA_FOLDER = "1Data/"
    _configKeys.BINNED_FOLDER = "2Binned/"
    _configKeys.STANDARDIZED_FOLDER = "3Standardized_Binned/"
    _configKeys.SUCCESSFULSTANDARDIZEDBINS = "3successfulStandardizedBins_2018.csv"

    pipelineP1b = __import__('1_stockImporting.py')
    pipelineP2b = __import__('2_binning.py')
    pipelineP3b = __import__('3_standardizing.py')

    successful_2016 = pd.read_csv("3successfulStandardizedBins_2016.csv", low_memory=False)
    successful_2018 = pd.read_csv("3successfulStandardizedBins_2018.csv", low_memory=False)

    for index in successful_2018.index:
        successful_2016_SymbolList = list(successful_2016.Symbol)
        name = successful_2018["Symbol"][index]
        asset_class = successful_2018["Type"][index]
        if name in successful_2016_SymbolList:
            successfulBins["Symbol"].append(name)
            successfulBins["Type"].append(asset_class)

    _configKeys.SUCCESSFULSTANDARDIZEDBINS = "3successfulStandardizedBins.csv"
    df = pd.DataFrame(successfulBins, columns = ["Symbol", "Type"])
    df.to_csv(_configKeys.SUCCESSFULSTANDARDIZEDBINS, index=False)

    _configKeys.STARTPULL = "03/01/2016"
    _configKeys.ENDPULL = "07/01/2018"
    _configKeys.FIRSTINDEX = "2016-01-03"
    _configKeys.LASTINDEX = "2018-01-07"
    _configKeys.DATA_FOLDER = "1Data_2016/"
    _configKeys.BINNED_FOLDER = "2Binned_2016/"
    _configKeys.STANDARDIZED_FOLDER = "3Standardized_Binned_2016/"

    listOfInterest = ["GMStock", "AAPLStock", "GOLDStock", "GoldCommodity", "CCLStock"]
    for symbol in listOfInterest:
        _configKeys.YVALUETICKER = symbol
        pipelineP4 = __import__('4_lassoRegression.py')

    _configKeys.STARTPULL = "14/01/2018"
    _configKeys.ENDPULL = "05/01/2020"
    _configKeys.FIRSTINDEX = "2018-01-14"
    _configKeys.LASTINDEX = "2020-01-05"
    _configKeys.DATA_FOLDER = "1Data/"
    _configKeys.BINNED_FOLDER = "2Binned/"
    _configKeys.STANDARDIZED_FOLDER = "3Standardized_Binned/"

    pipelineP5 = __import__('testing.py')

main()
# Psuedo code for the file:
# Run files 1-4 on 2016-2018 data.
# Run files 1-3 on 2018-2020 data.
# Run file 5 on 2018-2020 data.

# Things that this file needs to accomplish:
# All stock and pulled tickers need to be able to run on both 2016-2018 and 2018-2020 data.
# This means that any given stock pulled from 2016-2018 must be also standardizable in 2018-2020.

# Extra Psuedo code:
# Run files 1-3 on 2016-2018 data
# Run files 1-3 on 2018-2020 data
# Check 3successfulStandardizedBin.csv created from 2016-2018 and 2018-2020 data
# Delete all ticker that don't exists in both the 3successfulStandardizedBin.csv files
# Run file 4 on 2016-2018 data
# Run file 5 on 2018-2020 data
