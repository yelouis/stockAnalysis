# This will be the subMasterfile that runs the python files from 1-5.
import _configKeys
import pandas as pd
pipelineP1 = __import__('1_stockImporting')
pipelineP2 = __import__('2_binning')
pipelineP3 = __import__('3_standardizing')
pipelineP4 = __import__('4_lassoRegression')
pipelineP5 = __import__('5_testing')
pipelineP6 = __import__('6_paperTrading')

def main():

    # '''
    # Run importing and binning on 2016 data.
    # '''
    # _configKeys.STARTPULL = "03/01/2016"
    # _configKeys.ENDPULL = "07/01/2018"
    # _configKeys.FIRSTINDEX = "2016-01-03"
    # _configKeys.LASTINDEX = "2018-01-07"
    # _configKeys.DATA_FOLDER = "1Data_2016/"
    # _configKeys.BINNED_FOLDER = "2Binned_2016/"
    # _configKeys.SUCCESSFULWEEKBINS = "2successfulWeekBins2016.csv"
    # pipelineP1.main()
    # pipelineP2.main()
    #
    # '''
    # Run importing and binning on 2018 data
    # '''
    # _configKeys.STARTPULL = "14/01/2018"
    # _configKeys.ENDPULL = "05/01/2020"
    # _configKeys.FIRSTINDEX = "2018-01-14"
    # _configKeys.LASTINDEX = "2020-01-05"
    # _configKeys.DATA_FOLDER = "1Data/"
    # _configKeys.BINNED_FOLDER = "2Binned/"
    # _configKeys.SUCCESSFULWEEKBINS = "2successfulWeekBins2018.csv"
    # pipelineP1.main()
    # pipelineP2.main()

    '''
    Run the rest of the pipeline for a particular beta value
    '''
    betaList = [4, 6, 8, 12, 20]
    for betaValue in betaList:
        _configKeys.WINDOW_LENGTH = betaValue
        successfulBins = {"Symbol" : [], "Type" : []}

        '''
        Standardize 2016 data
        '''
        _configKeys.STARTPULL = "03/01/2016"
        _configKeys.ENDPULL = "07/01/2018"
        _configKeys.FIRSTINDEX = "2016-01-03"
        _configKeys.LASTINDEX = "2018-01-07"
        _configKeys.DATA_FOLDER = "1Data_2016/"
        _configKeys.BINNED_FOLDER = "2Binned_2016/"
        _configKeys.STANDARDIZED_FOLDER = "3Standardized_Binned_2016/"
        _configKeys.SUCCESSFULWEEKBINS = "2successfulWeekBins2016.csv"
        _configKeys.SUCCESSFULSTANDARDIZEDBINS = "3successfulStandardizedBins_2016.csv"
        pipelineP3.main()

        '''
        Standardize 2018 data
        '''
        _configKeys.STARTPULL = "14/01/2018"
        _configKeys.ENDPULL = "05/01/2020"
        _configKeys.FIRSTINDEX = "2018-01-14"
        _configKeys.LASTINDEX = "2020-01-05"
        _configKeys.DATA_FOLDER = "1Data/"
        _configKeys.BINNED_FOLDER = "2Binned/"
        _configKeys.STANDARDIZED_FOLDER = "3Standardized_Binned/"
        _configKeys.SUCCESSFULWEEKBINS = "2successfulWeekBins2018.csv"
        _configKeys.SUCCESSFULSTANDARDIZEDBINS = "3successfulStandardizedBins_2018.csv"
        pipelineP3.main()

        '''
        Saves all tickers that successfully standardized in 2016 and 2018 to a file called 3successfulStandardizedBins.csv
        '''
        successful_2016 = pd.read_csv("3successfulStandardizedBins_2016.csv", low_memory=False)
        successful_2018 = pd.read_csv("3successfulStandardizedBins_2018.csv", low_memory=False)

        for index in successful_2018.index:
            successful_2016_SymbolList = list(successful_2016.Symbol)
            name = successful_2018["Symbol"][index]
            asset_class = successful_2018["Type"][index]
            if name in successful_2016_SymbolList and name not in list(successfulBins["Symbol"]):
                successfulBins["Symbol"].append(name)
                successfulBins["Type"].append(asset_class)
        _configKeys.SUCCESSFULSTANDARDIZEDBINS = "3successfulStandardizedBins.csv"
        df = pd.DataFrame(successfulBins, columns = ["Symbol", "Type"])
        df.to_csv(_configKeys.SUCCESSFULSTANDARDIZEDBINS, index=False)

        '''
        Runs the rest of the pipeline on specfic tickers in listOfInterest
        '''

        # listOfInterest = ["MMMStock", "AXPStock", "AAPLStock", "BAStock", "CATStock", "CVXStock",
        #                 "CSCOStock", "KOStock", "DISStock", "XOMStock", "GSStock", "HDStock", "IBMStock",
        #                 "INTCStock", "JNJStock", "JPMStock", "MCDStock", "MRKStock", "MSFTStock", "NKEStock",
        #                 "PFEStock", "PGStock", "TRVStock", "UTXStock", "UNHStock", "VZStock", "VStock", "WMTStock",
        #                 "WBAStock"]

        listOfInterest = ["MMMStock", "AXPStock", "AAPLStock", "BAStock", "CATStock", "CVXStock",
                        "CSCOStock", "KOStock", "DISStock", "XOMStock", "GSStock", "HDStock", "IBMStock", "INTCStock", "JNJStock"] #Should take approx. 20 hours

        for symbol in listOfInterest:
            _configKeys.YVALUETICKER = symbol

            '''
            Creates lasso files using 2016 data
            '''
            _configKeys.STARTPULL = "03/01/2016"
            _configKeys.ENDPULL = "07/01/2018"
            _configKeys.FIRSTINDEX = "2016-01-03"
            _configKeys.LASTINDEX = "2018-01-07"
            _configKeys.DATA_FOLDER = "1Data_2016/"
            _configKeys.BINNED_FOLDER = "2Binned_2016/"
            _configKeys.STANDARDIZED_FOLDER = "3Standardized_Binned_2016/"
            pipelineP4.main()

            '''
            Creates testing files using 2018 data
            '''
            _configKeys.STARTPULL = "14/01/2018"
            _configKeys.ENDPULL = "05/01/2020"
            _configKeys.FIRSTINDEX = "2018-01-14"
            _configKeys.LASTINDEX = "2020-01-05"
            _configKeys.DATA_FOLDER = "1Data/"
            _configKeys.BINNED_FOLDER = "2Binned/"
            _configKeys.STANDARDIZED_FOLDER = "3Standardized_Binned/"
            pipelineP5.main()

            '''
            Run paperTrading.py
            '''
            pipelineP6.main()

main()
