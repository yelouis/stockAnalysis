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
