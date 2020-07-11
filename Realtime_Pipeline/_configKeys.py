import datetime

STARTPULL_DATES = ["14/01/2018", "21/01/2018", "28/01/2018", "04/02/2018", "11/02/2018", "18/02/2018", "25/02/2018", "04/03/2018", "11/03/2018", "18/03/2018", "25/03/2018", "01/04/2018", "08/04/2018", "15/04/2018"]
STARTINDEX_DATES = ["2018-01-14", "2018-01-21", "2018-01-28", "2018-02-04", "2018-02-11", "2018-02-18", "2018-02-25", "2018-03-04", "2018-03-11", "2018-03-18", "2018-03-25", "2018-04-01", "2018-04-08", "2018-04-15"]
ENDPULL_DATES = ["13/01/2019", "20/01/2019", "27/01/2019", "03/02/2019", "10/02/2019", "17/02/2019", "24/02/2019", "03/03/2019", "10/03/2019", "17/03/2019", "24/03/2019", "31/03/2019", "07/04/2019", "14/04/2019"]
ENDINDEX_DATES = ["2019-01-13", "2019-01-20", "2019-01-27", "2019-02-03", "2019-02-10", "2019-02-17", "2019-02-24", "2019-03-03", "2019-03-10", "2019-03-17", "2019-03-24", "2019-03-31", "2019-04-07", "2019-04-14"]

i = 0
STARTPULL = STARTPULL_DATES[i]
ENDPULL = STARTINDEX_DATES[i]
FIRSTINDEX = ENDPULL_DATES[i]
LASTINDEX = ENDINDEX_DATES[i]

DATA_FOLDER = "1Data/"
BINNED_FOLDER = "2Binned/"
STANDARDIZED_FOLDER = "3Standardized_Binned/"
SUCCESSFULSTANDARDIZEDBINS = "3successfulStandardizedBins.csv"
SUCCESSFULWEEKBINS = "2successfulWeekBins.csv"

LASSO_RESULTS_FOLDER = "4Lasso_Results"
TESTING_RESULTS_FOLDER = "5Testing_Results"
LIMIT_RESULTS_FOLDER = "6Limit_Results"
PROFIT_COMPARISONS_FOLDER = "7Profit_Comparisons"
WINDOW_LENGTH = 4
THRESHOLD = .1

YVALUETICKER = "BAStock"
SELL_LIMIT_MARKER = "High_average"
BUY_LIMIT_MARKER = "Low_average"
