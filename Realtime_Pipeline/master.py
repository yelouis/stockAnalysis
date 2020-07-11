
import datetime
import _configKeys
pipelineP1 = __import__('1_stockImporting')
pipelineP2 = __import__('2_binning')
pipelineP3 = __import__('3_standardizing')
pipelineP4 = __import__('4_lassoRegression')
pipelineP5 = __import__('5_createEstimation')
pipelineP6 = __import__('6_limitCreation')

'''
Pseudocode:
    - Nothing needs to be changed in pipeline 1-4
Pipeline 1-3 gets us to a standardized and binned format for our data

Need to optimize pipeline 5 so that it only creates an estimation using the most
recent window length of data.
    - Rename pipeline 5 to 5_createEstimation.py instead of 5_testing.py
    - Multiply coefficient from lasso results to just the last window length
        row of data from the standardizing output files

Need to optimize part 6 of the pipeline so I can just get the threshold to use.
    - Rename 6_paperTrading to 6_calculateLimits
    - Calculate how far off the predictions were from the actual value from the output of the 5_createEstimation file
        to create thresholds
    - Add those thresholds to the predict highs and lows value of the current week and return them as limits to
        our real time orders.

Have 8_alpacaTradingBot.py take in these limits and create orders for the week.
    - Find out how to use requests.get(ACCOUNT_URL, headers = HEADERS) to get day trading value

'''



def main():
    '''
    Run importing and binning on current data.
    '''
    
    pipelineP1.main()
    pipelineP2.main()
    pipelineP3.main()
    pipelineP4.main()
    pipelineP5.main()
    pipelineP6.main()


if __name__ == "__main__":
    main()
