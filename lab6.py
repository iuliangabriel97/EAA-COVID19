import csv
from math import sqrt

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import scipy
from scipy.stats import ttest_rel, ttest_ind, ttest_ind_from_stats

STANDARD_TRESHOLD_1 = 0.05
STANDARD_TRESHOLD_2 = 0.01


def problema1():
    data = []
    with open("background.csv", "r") as csv_file:
        reader = csv.reader(csv_file)
        for counter, row in enumerate(reader):
            if counter == 0:
                continue
            # print(row)
            data.append(row)

    data = [[entry[0], int(entry[1]), int(entry[2])] for entry in data]

    times = {0: None, 1: None}
    for color in times.keys():
        times[color] = [entry[2] for entry in data if entry[1] == color]

    for color in times.keys():
        _data = [[color, time] for time in times.keys()]
        data_frame = pd.DataFrame(times[color], columns=["Times"])
        print(data_frame.describe())

    # mean = np.mean(times[0])
    # times[0].append(mean)

    print(ttest_ind(times[0], times[1]))


def problema2():
    nobs1 = 18
    mean1 = 5.3
    std1 = 1.4

    nobs2 = 12
    mean2 = 4.8
    std2 = 1.6

    result = ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2)
    tvalue = result[0]
    pvalue = result[1]

    print(tvalue, pvalue)
    if not (pvalue < STANDARD_TRESHOLD_1 or pvalue < STANDARD_TRESHOLD_2):
        print("Resping ipoteza nula")
    else:
        print("Accept")


def problema3():
    nobs1 = 30
    mean1 = 6.7
    std1 = 0.6

    nobs2 = 20
    mean2 = 7.5
    std2 = 1.2

    result = ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2)
    tvalue = result[0]
    pvalue = result[1]

    print(tvalue, pvalue)
    if not (pvalue < STANDARD_TRESHOLD_1 or pvalue < STANDARD_TRESHOLD_2):
        print("Resping ipoteza nula")
    else:
        print("Accept")


if __name__ == '__main__':
    # problema1()
    # problema2()
    problema3()
