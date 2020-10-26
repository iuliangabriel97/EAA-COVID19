import operator

import numpy as np
import docplex.mp.model as cpx
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
from scipy.stats import expon


# U = people
# V = facilites
# p : U x V -> [0,1] = percentage of time spent by a person in a facility
# f : U -> [0,1]  = probability to get infected
# c : U -> R+ = closure cost of facility
# c': U -> R+ = cost of isolating people
# B = buget

# xu = 0|1 depending if a person is isolated or not, for each person in U
# yv = 0|1 depending if a facility is closed or not, for each facility in V
# ru = the risk of a person
# RU = the risk of a facility

def power_law(k_min, k_max, y, gamma):
    return ((k_max ** (-gamma + 1) - k_min ** (-gamma + 1)) * y + k_min ** (-gamma + 1.0)) ** (1.0 / (-gamma + 1.0))


TESTS = 1

# Number of facilities
Facilities = 500

# Minimum size of a company
k_min = 4

# Maximum size of a company
k_max = 1000

# Power law exponent for the size of the factories
gamma = 1.1

# Average number of activities per day
activities = 4

# Power law exponent for the infection chance of people
gamma2 = 10

# The percentage of the total budget
how_much_budget = 0.1

# How many people can be quarantined with our given budget
frac_people = 0.01

# Cost of the facilities
norm_mean = 1.1
norm_std = 0.4

# We divide the available budget
dB = 0.2

for tests in range(TESTS):
    print(f"Test no. {tests}")
    # Distributing people to facilities
    size_fac = np.zeros(Facilities, int)
    y = np.zeros(Facilities, float)
    for n in range(Facilities):
        y[n] = np.random.uniform(0, 1)
        size_fac[n] = int(power_law(k_min, k_max, y[n], gamma))

    # plt.plot(range(1, Facilities + 1), sorted(size_fac, reverse=True), 'ro')
    # plt.show()

    # this is the total number of visits to facilities
    Total_visitors = int(sum(size_fac))
    print("Total visitors " + str(Total_visitors))

    # Population size
    People = int(Total_visitors / activities)
    print("Population " + str(People))

    # Creating the graph
    G = nx.Graph()

    # People are numbered from 0 to People and Facilities from People + 1 to People + Facilities -1
    nodes = list(range(People)) + list(map(lambda x: x + People, range(Facilities)))
    G.add_nodes_from(nodes)

    neighbor = list([] for i in range(People))

    count = np.zeros(People, int)
    adj_list = []
    for i in range(Facilities):
        adj_list.append(list(np.random.randint(0, People, size_fac[i])))
        # print(adj_list[i])
        for j in adj_list[i]:
            count[j] = count[j] + 1
            neighbor[j].append(i)
            G.add_edge(j, i + People)

    for i in range(People):
        data_expon = expon.rvs(scale=6, loc=1, size=count[i])
        z = sum(data_expon)
        data_norm = list(map(lambda x: float(x / z), data_expon))

        k = 0
        for j in neighbor[i]:
            G.add_edge(i, People + j, weight=data_norm[k])
            k = k + 1

    # print(f"Graph is bipartite {nx.is_bipartite(G)}")

    f = np.zeros(People)
    for n in range(People):
        f[n] = power_law(0.01, 0.99, np.random.uniform(0, 1), gamma2)

    adjacency_matrix = []
    for u in range(People):
        new_row = list()
        for v in range(People, People + Facilities):
            value = G.get_edge_data(u, v)
            value = 0 if value is None else value["weight"]
            new_row.append(value)
            # print(f"a[{u}][{v}] = {G.get_edge_data(u, v, 0)}")
        adjacency_matrix.append(new_row)

    # print(adjacency_matrix)
    # for u in range(People):
    #     for v in range(Facilities):
    #         print(f"a[{u}][{v}] = {adjacency_matrix[u][v]}")

    # Cost of the facilities
    cost = np.zeros(Facilities)
    for i in range(Facilities):
        # select poisson variable
        x = np.random.normal(norm_mean, norm_std)
        cost[i] = size_fac[i] ** x

    B = how_much_budget * sum(cost)

    # aici e buba
    # Cost of the people
    norm_std_cost_people = 0.4
    norm_mean_cost_people = 1.1
    cost_prime = np.zeros(People)
    cost_universal_people = B / People
    a = B / People * 1 / frac_people

    for i in range(People):
        # select poisson variable
        x = np.random.normal(norm_mean_cost_people, norm_std_cost_people)
        cost_prime[i] = x

    cost_prime = np.zeros(People)
    for i in range(People):
        for j in range(Facilities):
            cost_prime[j] += adjacency_matrix[i][j]

    print(B, sum(cost), sum(cost_prime))
    # for u in range(People):
    #     print(cost_prime[u])

    # Calculating the risk of a facility
    R = np.zeros(Facilities)
    for v in range(Facilities):
        for u in range(People):
            R[v] = R[v] + f[u] * adjacency_matrix[u][v]

    # Calculating the risk of the people
    r = np.zeros(People)
    for u in range(People):
        for v in range(Facilities):
            r[u] = r[u] + R[v] * adjacency_matrix[u][v]

    model = cpx.Model("COVID-19 MODEL")
    model.context.cplex_parameters.threads = 128

    x = {i: model.binary_var(name=f"x{i}") for i in range(People)}
    y = {i: model.binary_var(name=f"y{i}") for i in range(Facilities)}

    # print(f"B = {B}\nr = {r}\n\nf = {f}\nR = {R}\nx = {x}\ny = {y}")

    objective = model.sum(r[u] * x[u] for u in range(People))
    model.minimize(objective)

    model.float_precision = 8
    model.add_constraint(model.sum(cost_prime[u] * x[u] for u in range(People)) + model.sum(
        cost[v] * y[v] for v in range(Facilities)) >= B * 0.8)
    model.add_constraint(model.sum(cost_prime[u] * x[u] for u in range(People)) + model.sum(
        cost[v] * y[v] for v in range(Facilities)) <= B * 0.99)

    model.add_constraints([
        model.sum(f[u] * adjacency_matrix[u][v] * x[u] for u in range(People)) <= R[v] for v in range(Facilities)])
    model.add_constraints([
        model.sum(adjacency_matrix[u][v] * R[v] * y[v] for v in range(Facilities)) <= r[u] for u in range(People)])

    solution = model.solve(log_output=True)
    model.print_solution(print_zeros=False)
    model.print_information()
    print(solution)
