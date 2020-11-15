import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

data_file_path = "EDA3.dat"


def read_file(filepath):
    data = []
    with open(filepath) as fd:
        for counter, line in enumerate(fd):
            if counter == 0:
                continue
            line = line.strip()
            if not line:
                continue
            _id, age, agegroup, strength, sex, party = line.split(" ")
            data.append({"id": _id, "age": int(age), "agegroup": agegroup, "strength": int(strength), "sex": sex,
                         "party": party})

    return data


if __name__ == '__main__':
    data = read_file(data_file_path)
    _data = [[entry["age"], entry["strength"]] for entry in data]
    data_frame = pd.DataFrame(_data, columns=["Age", "Strength"])
    print(data_frame.describe())
    # boxplot = data_frame.boxplot()
    # plt.show()
    print(f"Covariance:\n{data_frame.cov()}")
    print(f"Correlation:\n{data_frame.corr()}")
    # heatmap = sns.heatmap(data_frame)
    # plt.show()
    scatter_plot = plt.scatter(x=[item[0] for item in _data], y=[item[1] for item in _data])
    plt.show()

    # fig, ax = plt.subplots()
    # ages = np.array([entry["age"] for entry in data])
    # strengths = np.array([entry["strength"] for entry in data])
    # parties = np.array([entry["party"] for entry in data])
    # for p in np.unique(parties):
    #     i = np.where(parties == p)
    #     ax.scatter(strengths[i], ages[i], label=p)
    # ax.legend()
    # plt.show()
