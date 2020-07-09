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

#Finds the best csv for predictions
def findCSV():
    successfulTestingDF = pd.read_csv("5successfulTesting.csv", low_memory=False)
    testingCSV = list(successfulTestingDF["FileName"].values)[0]
    # Having the [0] means that we are only looking at the first value in the 5successfulTesting file
    # This is under the assumption that since we only pick the best alpha from a selected beta,
    #    we will only have one file to calculate limits on
    print(testingCSV)
    quit()
    return testingCSV

#def calculateThreshold():

def main():
    findCSV()
    quit()

if __name__ == "__main__":
    main()
