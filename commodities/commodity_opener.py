#pip install quandl

import quandl
import os
import pandas as pd

quandl.ApiConfig.api_key = '5ypcn9Wtm8mxHBuLmwkf'

df = pd.read_csv("commodity_tickers.csv", low_memory=False)

for ind in df.index:

    if str(df['Source'][ind]) != "ODA":
        continue

    fileName = str(df['Code'][ind])

    mydata = quandl.get([fileName + ".1", fileName + ".2"], start_date="1995-01-01", end_date="2020-03-01")

    commodity_name = str(df['Name'][ind])
    charactersToRemove = [' ', ',']
    for char in charactersToRemove:
        commodity_name = commodity_name.replace(char, "")

    mydata.to_csv(os.path.join(os.getcwd(), 'commodity_CSV', commodity_name + '.csv'))
