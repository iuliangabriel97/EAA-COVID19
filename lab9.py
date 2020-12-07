import pandas as pd
import numpy as np
import scipy.stats as stats

signifiance_level = 0.05


def problema1():
    m = 20
    n = 25
    k1 = 6
    k2 = 19
    p1 = stats.binom_test(x=k1, n=n, p=0.5)
    p2 = stats.binom_test(x=k2, n=n, p=0.5)
    print(p1, p2)
    P = 2 * min(p1, p2)
    if P < signifiance_level:
        print("The median response time for this printer DOES NOT exceed 20 sec")
    else:
        print("The median response time for this printer exceeds 20 sec")



def problema2():
    m = 50
    salaries1 = [47, 52, 68, 72, 55, 44, 58, 63, 54, 59, 77]
    salaries2 = [m for m in range(len(salaries1))]
    diff = [salaries1[i] - salaries2[i] for i in range(len(salaries1))]
    u_statistic, p_value = stats.wilcoxon(diff)
    if p_value < signifiance_level:
        print(
            "The test DOES NOT provide significant evidence that the median starting salary of software developers is above 50, 000$")
    else:
        print(
            "The test provides significant evidence that the median starting salary of software developers is above 50, 000$")


def problema3():
    cost1 = [89, 99, 119, 139, 189, 199, 229]
    cost2 = [109, 159, 179, 209, 219, 259, 279, 299, 309]
    u_statistic, p_value = stats.mannwhitneyu(cost1, cost2)
    if p_value < signifiance_level:
        print("The median cost is decreasing")
    else:
        print("The median cost in rising")


def ex1():
    n = 20
    # Buisness Volume time 0
    bv0 = np.random.normal(loc=3, scale=.1, size=n)
    # Buisness Volume time 1
    bv1 = bv0 + 0.1 + np.random.normal(loc=0, scale=.1, size=n)
    # create an outlier
    bv1[0] -= 10
    # Paired t-
    print(stats.ttest_rel(bv0, bv1))
    # Wilcoxon
    print(stats.wilcoxon(bv0, bv1))

def ex2():
    n = 20
    # Buismess Volume group 0
    bv0 = np.random.normal(loc=1, scale=.1, size=n)
    # Buismess Volume group 1
    bv1 = np.random.normal(loc=1.2, scale=.1, size=n)
    # create an outlier
    bv1[0] -= 10
    # Two-samples t-test
    print(stats.ttest_ind(bv0, bv1))
    # Wilcoxon
    print(stats.mannwhitneyu(bv0, bv1))

def problema4():
    ex1()
    ex2()


def main():
    problema1()
    # problema2()
    # problema3()
    # problema4()


if __name__ == '__main__':
    main()
