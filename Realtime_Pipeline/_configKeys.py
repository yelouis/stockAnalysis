import datetime


STARTPULL = "14/01/2018"
ENDPULL = datetime.datetime.strftime(datetime.datetime.date(datetime.datetime.now()),'%d/%m/%Y')
FIRSTINDEX = "2018-01-14"
LASTINDEX = datetime.datetime.strftime(datetime.datetime.date(datetime.datetime.now()),'%Y-%m-%d')

DATA_FOLDER = "1Data/"
BINNED_FOLDER = "2Binned/"
STANDARDIZED_FOLDER = "3Standardized_Binned/"
SUCCESSFULSTANDARDIZEDBINS = "3successfulStandardizedBins.csv"
SUCCESSFULWEEKBINS = "2successfulWeekBins.csv"

LASSO_RESULTS_FOLDER = "4Lasso_Results"
TESTING_RESULTS_FOLDER = "5Testing_Results"
LIMIT_RESULTS_FOLDER = "6Limit_Results"
PROFIT_COMPARISONS_FOLDER = "7Profit_Comparisons"
WINDOW_LENGTH = 12
THRESHOLD = .1

YVALUETICKER = "JNJStock"
SELL_LIMIT_MARKER = "High_average"
BUY_LIMIT_MARKER = "Low_average"
