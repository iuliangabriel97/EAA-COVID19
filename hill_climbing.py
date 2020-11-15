import json
import random

import networkx as nx
import numpy as np
from scipy.stats import expon
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def power_law(k_min, k_max, y, gamma):
    return ((k_max ** (-gamma + 1) - k_min ** (-gamma + 1)) * y + k_min ** (-gamma + 1.0)) ** (1.0 / (-gamma + 1.0))


TESTS = 1

# Number of facilities
Facilities = 50

# Minimum size of a company
k_min = 4

# Maximum size of a company
k_max = 100

# Power law exponent for the size of the factories
gamma = 1.1

# Average number of activities per day
activities = 4

# Power law exponent for the infection chance of people
gamma2 = 10

# The percentage of the total budget
how_much_budget = 0.01

# How many people can be quarantined with our given budget
frac_people = 0.01

# Cost of the facilities
norm_mean = 1.1
norm_std = 0.4

# We divide the available budget
dB = 0.2

my_plot_rules = []

for dB in np.arange(0, 1.01, 0.05):
    avg_risk = 0
    max_risk = 0
    min_risk = 10000

    for tests in range(TESTS):

        # Distributing people to facilities
        size_fac = np.zeros(Facilities, int)
        y = np.zeros(Facilities, float)
        for n in range(Facilities):
            y[n] = np.random.uniform(0, 1)
            size_fac[n] = int(power_law(k_min, k_max, y[n], gamma))

        # plt.plot(range(1,Facilities+1),sorted(size_fac,reverse=True),'ro')
        # plt.show()

        # exit()

        # this is the total number of visits to facilities
        Total_visitors = int(sum(size_fac))
        # print("Total visitors " + str(Total_visitors))

        # Population size
        People = int(Total_visitors / activities)
        # print("Population " + str(People))

        # Creating the graph
        G = nx.Graph()

        # People are numbered from 0 to People and Facilities from People + 1 to People + Facilities -1
        nodes = list(range(People)) + list(map(lambda x: x + People, range(Facilities)))
        G.add_nodes_from(nodes)
        # print(nodes)

        neighbor = list([] for i in range(People))

        count = np.zeros(People, int)
        adj_list = []
        for i in range(Facilities):
            adj_list.append(list(np.random.randint(0, People, size_fac[i])))
            # print(adj_list[i])
            for j in adj_list[i]:
                count[j] = count[j] + 1
                neighbor[j].append(i)
                # G.add_edge(j,i+People)

        # nx.draw(G)
        ### Sanity check 1 ###

        pois = np.zeros(int(max(count)) + 1, int)
        for i in range(People):
            pois[int(count[i])] = pois[int(count[i])] + 1

        # print (sorted(pois))
        # plt.plot(range(int(max(count))+1),pois,'ro')
        # plt.show()

        #### Sanity check 1 End: We have indeed poisson distribution

        for i in range(People):
            data_expon = expon.rvs(scale=6, loc=1, size=count[i])
            z = sum(data_expon)
            data_norm = list(map(lambda x: float(x / z), data_expon))
            # print(len(data_norm))
            # print(len(neighbor[i]))
            # print(count[i])
            # plt.plot(range(count[i]),sorted(data_norm,reverse=True),'ro')
            # plt.show()

            k = 0
            # print(data_norm)
            for j in neighbor[i]:
                G.add_edge(i, People + j, weight=data_norm[k])
                k = k + 1

        # print(nx.is_bipartite(G))
        # nx.draw(G)

        f = np.zeros(People)
        for n in range(People):
            f[n] = power_law(0.01, 0.99, np.random.uniform(0, 1), gamma2)

        # Calculating the risk of a facility
        R = np.zeros(Facilities)
        for i in range(Facilities):
            for j in G.edges(People + i, data='weight'):
                R[i] = R[i] + f[j[1]] * j[2]

        # plt.plot(scale_free_distribution,R,'ro')
        # plt.show()

        # Calculating the risk of the people
        r = np.zeros(People)
        for i in range(People):
            for j in G.edges(i, data='weight'):
                r[i] = r[i] + R[j[1] - People] * j[2]

        # plt.plot(range(People),sorted(r),'ro')
        # plt.show()

        OriginalRisk = sum(r)

        # Cost of the facilities
        cost = np.zeros(Facilities)
        for i in range(Facilities):
            # select poisson variable
            x = np.random.normal(norm_mean, norm_std)
            cost[i] = size_fac[i] ** x

        # plt.plot(size_fac,cost,'ro')
        # plt.show()

        efficiency = np.zeros((Facilities, 2))
        for i in range(Facilities):
            efficiency[i][0] = i
            efficiency[i][1] = cost[i] / R[i]

        # print (sorted(efficiency,key = operator.itemgetter(1)))

        # set the the budget
        B = how_much_budget * sum(cost)

        # set the parameters for quaranteening people
        a = B / People * 1 / frac_people
        b = 0

        # print("Budget " + str(B))
        # print("Quarantine budget " + str(dB*B))
        # print(cost)
        # list of quarantined people
        qp = []

        # list of closed facilities
        cf = []


def evaluate_solution(sol):
    total_cost = 0
    for element in sol:
        # total_cost += cost[element]
        total_cost += efficiency[element][1]

    return total_cost


def generate_random_solution():
    return random.choices(population=list(range(Facilities)), k=1)


def mutate_solution(solution, already_picked):
    # print(f"already picked = {already_picked}")
    random_pick = random.choice(list(set(range(Facilities)) - set(already_picked)))
    already_picked.add(random_pick)
    solution.append(random_pick)


def calculate_cost(sol):
    s = 0
    for element in sol:
        s += cost[element]

    return s


budget = B
global_best_score = float('-inf')
global_best_solution = None
global_total_closure_cost = None
solutions_dict = {}
for i in range(30):
    already_picked_facilities = set()
    total_closure_cost = 0
    solution = generate_random_solution()
    best_score = evaluate_solution(solution)
    total_closure_cost += calculate_cost(solution)
    valid_sol = False
    while total_closure_cost <= budget:
        valid_sol = True
        # print(f"Best cost so far {best_score} \n Solution {solution}")
        new_solution = solution.copy()
        mutate_solution(new_solution, already_picked_facilities)

        score = evaluate_solution(new_solution)
        # print(f"score ={score} best_Score={best_score}")
        c = calculate_cost(new_solution)
        if total_closure_cost + c > budget:
            break

        if score > best_score:
            best_score = score
            solution = new_solution
            total_closure_cost += c

    solutions_dict[i] = {"effiency": best_score, "solution": solution, "closure_cost": calculate_cost(solution)}


    if best_score > global_best_score and valid_sol:
        global_best_score = best_score
        global_best_solution = solution
        global_total_closure_cost = calculate_cost(solution)

print(f"Budget {budget}\n Cost {global_total_closure_cost}")
print(global_best_solution)

data_frame_risk = pd.DataFrame(R, columns=["Risk"])
result = data_frame_risk.describe()
print(result)
risk_boxplot = data_frame_risk.boxplot()
plt.show()

data_frame_cost = pd.DataFrame(cost, columns=["Cost"])
result = data_frame_cost.describe()
print(result)
cost_boxplot = data_frame_cost.boxplot()
plt.show()

scatter_plot = plt.scatter(x=cost, y=R)
plt.show()

solutions_dict["budget"] = budget
solutions_dict["best_solution"] = {"cost": global_total_closure_cost, "effiency": global_best_score,
                                   "solution": global_best_solution}

with open("covid-19-hill-climbing.json", "w") as fd:
    json.dump(solutions_dict, fd, indent=4)
