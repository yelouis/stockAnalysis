# This will be the subMasterfile that runs the python files from 1-5.

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
