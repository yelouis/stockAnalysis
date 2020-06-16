#This file will keep track of all the moving variables and we can slowly add to that file
import _configKeys

import pandas as pd
import collections
import copy
import os
import csv
import datetime
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import statistics
from pathlib import Path
import time
import numpy as np
import copy
import math
from sklearn.metrics import mean_absolute_error


def readSuccessfulFile(reference_df):
    referenceDict = {}
    for index in reference_df.index:
        name = reference_df["Symbol"][index]
        asset_class = reference_df["Type"][index]
        referenceDict[name] = asset_class
    return referenceDict

def main():
    reference_df = pd.read_csv("3successfulStandardizedBins.csv", low_memory=False)
    referenceDict = readSuccessfulFile(reference_df)

    (xValues, xValueNames) = readXValues(referenceDict, "")

    xValues = list(map(list, zip(*xValues)))

    #YValueDict is organized such that the key is the statistic that we are interested
    #in and the value is a list containing the yValues.
    yValueDict = readYValues()

    for yValueKey in yValueDict:
        something()


main()

'''
pseudocode:

initialize a total prediction list with all 0's
for every Feature Name in the Feature Name column:
    1) check the coefficient to see if its absolut value is over a threshhold (0.1?)
    2) if above the threshhold: (ex: ALIM_Volume_average | 1.3)
        Open the ALIM.csv file from the standardized folder
        get the element_statistic column (Volume_average) and index from [0:length - 1]
        multiply this column's values by the coefficient (1.3) and add them to the total prediction list
add the total prediction list to a data frame (noting the value it is predicting and the fact that it is the prediction)
add the actual standardized values of the value we are trying to predict

Note: yValueName consists of yStock_Element_Statistic

df1:
{Date: [List of Dates (Yavlues so it wil probably start with 4/3/16, aka index 1)]
 'yValueName'_Predicted_Standardized: [total prediction list]
 'yValueName'_Actual_Standardized: [(Go to yStock file and find the matching yStock_Element_Statistic)[1:end] (in folder 3)]
}

We want to get df2 from df1:
{Date: [List of Dates (Yavlues so it wil probably start with 4/3/16, aka index 1)]
 'yValueName'_Predicted: [total prediction list (Unstandardized)]
 'yValueName'_Actual: [(Go to yStock file and find the matching yStock_Element_Statistic)[window_length(beta) + 1:end] (in folder 2)]
}

To do this, we need to use the unstandardizing function.

Final step:
Initialize a df3 that is empty

concatnate all df2s into df3

Write df3 into an csv with the name: 'yStock'_alpha_beta.csv @ location 5testing_Results
Example: Gold0.3_alpha13_beta.csv @ location 5testing_Results
Really it is the just the same name as the file you read in to get the coefficient

'''


def Calculate_Standardized_Value(series_to_standardize, window_length):
    return standardizeSeries([series_to_standardize], window_length)[0][0]

#We could try to get the math better, but it is very challenging when we are using the expected value in the standardization
# ((standardized_value * stdev) + (sum(known_values)/window_length)) / ((window_length-1)/window_length)
# (prediction[i] * stdev)
def Estimate_Unstandardized(standardized_value, known_values, window_length):

    '''
    known_values: the beta-1 weeks before the week of interest [$12, $14, $13.4, ...]
    window_length: the beta value
    standardized_value: the standardized value of the week we are trying to predict

    alg: [$12, $14, $13.4, ...] append x (estimated_value) variable
    keep changing x to get the predicted standardized value as close to the given standardized_value

    x - mean([$12, $14, $13.4, ... x])/ stdev([$12, $14, $13.4, ... x])
    Needs to get as close as possible to standardized_value
    '''

    estimated_value = int(known_values[-1])

    factor = 10 ** (len(str(int(known_values[-1]))) - 1) # Take most recent value as judge of where the prediction could move (+ or - 100% max)

    while(True):

        if factor < 0.001: # done when we know the nearest tenth of a cent
            break

        dif = standardized_value - Calculate_Standardized_Value(known_values + [estimated_value], window_length)

        if dif <= 0:
            for j in range(10):
                if abs(dif) > abs(standardized_value - Calculate_Standardized_Value(known_values + [estimated_value - factor], window_length)):
                    estimated_value -= factor
                    dif = standardized_value - Calculate_Standardized_Value(known_values + [estimated_value], window_length) # update dif
                else:
                    break
        else:
            for j in range(10):
                if abs(dif) > abs(standardized_value - Calculate_Standardized_Value(known_values + [estimated_value + factor], window_length)):
                    estimated_value += factor
                    dif = standardized_value - Calculate_Standardized_Value(known_values + [estimated_value], window_length) # update dif
                else:
                    break

        factor = factor / 10

    return estimated_value
