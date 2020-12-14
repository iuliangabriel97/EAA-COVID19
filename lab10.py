import pandas as pd
import scipy.stats
import statsmodels.stats.multicomp
import scikit_posthocs as sp
import numpy as np

drug_a = [30, 35, 40, 25, 35]
drug_b = [25, 20, 30, 25, 30]
drug_c = [15, 20, 25, 20, 20]

P_VALUE = 0.05


def _1a():
    statistic, pvalue = scipy.stats.f_oneway(drug_a, drug_b, drug_c)
    print(f"One Way Anova - Reject the Null hypothesis {pvalue < P_VALUE}")


def _1b():
    result = statsmodels.stats.multicomp.pairwise_tukeyhsd([drug_a, drug_b, drug_c], ["drug_a", "drug_b", "drug_c"])
    print(f"Pairwaise Tukey HSD - Reject the Null hypothesis {result.reject}")


def _1c():
    x = pd.DataFrame({"drug_a": drug_a, "drug_b": drug_b, "drug_c": drug_c})
    x = x.melt(var_name="groups", value_name="values")
    result = sp.posthoc_scheffe(x, val_col="values", group_col="groups")
    print(result)


def ex1():
    # _1a()
    # _1b()
    _1c()


def ex2():
    metaheuristic1 = []
    metaheuristic2 = []
    metaheuristic3 = []
    metaheuristic4 = []
    with open(r"lab10_data.txt") as fd:
        for line in fd:
            line = line.strip()
            if not line:
                continue
            elements = line.split()
            metaheuristic1.append(elements[1])
            metaheuristic2.append(elements[4])
            metaheuristic3.append(elements[7])
            metaheuristic4.append(elements[10])

    stat, p = scipy.stats.friedmanchisquare(metaheuristic1, metaheuristic2, metaheuristic3, metaheuristic4)
    print(f"Friedman Test - Reject the null hypothesis {p < P_VALUE}")


def main():
    ex1()
    ex2()


if __name__ == '__main__':
    main()
