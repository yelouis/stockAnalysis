'''
This will be the master file that runs everything.

Create new folders liek 1Data_RealTime to allow testing to still work on older folders

Run files 1-3 on 2018-now (last Sunday) data

Pick a stock/etf/commodity that we are permanently going to stick with.

Run a file that tells us what the best alpha and beta values to use

Run that stock/etf/commodity in file 4-5

Run the papertrading file and have it return:
    - Buy price, sell price.
        - Buy Price = Lasso Prediction (Average Max) + - some threshold
        - Sell Price = Lasso Prediction (Average Min) + - some threshold
    - Save buy and sell price in a csv file.

Pass the buy and sell price to file called (realTimeAlphca) during the trading hours.
Create a transaction and portfolio class to keep track of the trades made by realTimeAlphca
and save that to a csv.
'''

'''
Remind threshold folks that they are using training data now and that is ok.
They should be trying to find the best threshold within the last 13 weeks (or window_length).
    - Therefore, they should only use data from the last 13 weeks from now, to calculate threshold
'''
import datetime
import _configKeys
pipelineP1 = __import__('1_stockImporting')
pipelineP2 = __import__('2_binning')
pipelineP3 = __import__('3_standardizing')
pipelineP4 = __import__('4_lassoRegression')
pipelineP5 = __import__('5_testing')
pipelineP6 = __import__('6_paperTrading')



def main():
    '''
    Run importing and binning on current data.
    '''

    _configKeys.STARTPULL = "14/01/2018"
    _configKeys.ENDPULL = datetime.datetime.strftime(datetime.datetime.date(datetime.datetime.now()),'%d/%m/%Y')
    _configKeys.FIRSTINDEX = "2018-01-14"
    _configKeys.LASTINDEX = datetime.datetime.strftime(datetime.datetime.date(datetime.datetime.now()),'%Y-%m-%d')
    _configKeys.DATA_FOLDER = "1Data/"
    _configKeys.BINNED_FOLDER = "2Binned/"
    _configKeys.SUCCESSFULWEEKBINS = "2successfulWeekBinsReal.csv"
    _configKeys.STANDARDIZED_FOLDER = "3Standardized_Binned/"
    _configKeys.SUCCESSFULSTANDARDIZEDBINS = "3successfulStandardizedBins_Real.csv"
    _configKeys.YVALUETICKER = symbol #we need to actually pick something to trade

    pipelineP1.main()
    pipelineP2.main()
    pipelineP3.main()
    pipelineP4.main()
    pipelineP5.main()


if __name__ == "__main__":
    main()
