import random
import numpy as np

import oapackage
from tqdm import tqdm



def reformat_graph(A):
    new_A = []
    for row in A:
        new_A.append([col if col != 0 else float('inf') for col in row])

    return new_A


def return_min_path(graph, source):
    dist = {}
    prev = {}
    Q = []
    for idv, v in enumerate(graph):
        dist[idv] = float('inf')
        prev[idv] = None
        Q.append(idv)
    dist[source] = 0

    while Q:
        dist_in_Q = {k:v for k, v in dist.items() if k in Q}
        u = min(dist_in_Q, key=dist_in_Q.get)
        Q.remove(u)
        for idc, col in enumerate(graph[u]):
            if idc in Q:
                alt = dist[u] + col

                if alt < dist[idc]:
                    dist[idc] = alt
                    prev[idc] = u
    return {'costs': dist, 'Prev':prev}



def generate_lower_triangular(number_nodes, mu_edge, std_edges,  mu_pol, sigma_pol, polarity_mapping, max_pol=1, min_pol=-1):

    edges_numbers = []
    polarities = []
    for x in np.random.normal(mu_edge, std_edges, number_nodes).tolist():
        if x <= 0:
            edges_numbers.append(1)
        elif x > number_nodes:
            edges_numbers.append(number_nodes)
        else:
            edges_numbers.append(int(round(x)))

    for x in np.random.normal(mu_pol, sigma_pol, number_nodes).tolist():
        if round(x) > max_pol:
            polarities.append(max_pol)
        elif round(x) < min_pol:
            polarities.append(min_pol)
        else:
            polarities.append(int(round(x)))


    nodes = [1, float('inf')]
    adj = np.matrix(np.ones((number_nodes, number_nodes)) * np.inf)
    new_matrix = []

    for idrow, row in enumerate(adj):
        tmp_row = row.tolist()[0]
        if idrow != 0:
            possible_values = list(range(idrow))
            random.shuffle(possible_values)
            n_edges = min(len(possible_values), edges_numbers[idrow])
            for id_edge in range(n_edges):
                tmp_row[possible_values[id_edge]] = polarity_mapping[polarities[idrow]]
        new_matrix.append(tmp_row)
    return new_matrix


def build_adjiacency_matrix(number_nodes, edges, polarities, pol_mapping):
    adj =  np.matrix(np.ones((number_nodes,number_nodes)) * np.inf)

    for edge in edges:
        adj[edge[0]-1, edge[1]-1] = pol_mapping[polarities[edge[-1]]]
    return adj

def get_paths(prev, targets):
    paths = {}
    for x in targets:
        seq = []
        u = x
        while u != None:
            seq.insert(0,u)
            u = prev[u]
        paths[x] = seq
    return paths

def uniform_costs(matrix):
    n_nodes = len(matrix)
    adj =  np.matrix(np.ones((n_nodes,n_nodes)) * np.inf)

    for r_id, r in enumerate(matrix):
        for c_id, c in enumerate(r):
            if c != np.inf:
                adj[r_id, c_id] = 1

    return adj.tolist()

def get_max_at_min(lengths, costs):
    max_len = max(lengths)
    new_l = []
    new_c = []
    for idx, x in enumerate(lengths):
        if x == max_len:
            new_l.append(x)
            new_c.append(costs[idx])
    id_best = new_c.index(min(new_c))

    return new_l[id_best], min(new_c)

def get_min_at_max(lengths, costs):
    min_cost = min(costs)
    new_l = []
    new_c = []
    for idx, x in enumerate(costs):
        if x == min_cost:
            new_c.append(x)
            new_l.append(lengths[idx])

    id_best = new_l.index(max(new_l))
    
    return new_c[id_best], max(new_l)

random.seed(123)  
np.random.seed(123)
a = [0]*60 + [-1]*25 + [1]*15
n_nodes = 14
mu_pol = np.array(a).mean()
sigma_pol = np.asarray(a).std()

mu_edge = 3
std_edges = 1

global_costs = []
global_lengths = []
max_erdos_numbers = []
pol_mapping = {1:2, 0:1, -1:0}

pol_mapping_uniform = {1:1, 0:1, -1:1}
max_lengths = []
max_lengths_cost =[]
min_costs = []
min_costs_length = []

for x in tqdm(range(100)):

    A = generate_lower_triangular(n_nodes, mu_edge, std_edges,  mu_pol, sigma_pol, pol_mapping)

    A_uniform = uniform_costs(A)
    source = n_nodes - 1

    res = return_min_path(A, source)
    res_uniform = return_min_path(A_uniform, source)
    
    targets = list(range(0, n_nodes))
    # print(targets)
    targets.remove(source)

    paths = get_paths(res['Prev'],targets)
    paths_uniform =  get_paths(res_uniform['Prev'],targets)

    len_paths = np.asarray([len(path) for node, path in sorted(paths.items())])


    len_paths_uniform = np.asarray([len(path) for node, path in sorted(paths_uniform.items())])
    max_len_uniform = len_paths_uniform.max()
    
    max_erdos_numbers.append(max_len_uniform)

    pareto=oapackage.ParetoDoubleLong()

    costs = [res['costs'][id_node+1] * -1  for id_node, length in enumerate(len_paths) if res['Prev'][id_node+1] != None]

    new_len_paths = [length for id_node, length in enumerate(len_paths) if res['Prev'][id_node+1] != None ]
   
    if len(new_len_paths) != 0:
  
        max_len = max(new_len_paths)
        index = new_len_paths.index(max_len)
        
        opp_cost = [x * -1 for x in costs]
       
        
        l1, c1 = get_max_at_min(new_len_paths, opp_cost)
        c2, l2 = get_min_at_max(new_len_paths, opp_cost)

        max_lengths.append(l1)
        max_lengths_cost.append(c1)

        min_costs.append(c2)
        min_costs_length.append(l2)

        datapoints = np.asanyarray([new_len_paths, costs ])

        for ii in range(0, datapoints.shape[1]):
            w = oapackage.doubleVector( (float(datapoints[0,ii]), float(datapoints[1,ii])))
            pareto.addvalue(w, ii)


        lst=pareto.allindices()
        optimal_datapoints=datapoints[:,lst]

        global_lengths.append(optimal_datapoints[0])
        global_costs.append(optimal_datapoints[1]*-1)

lengths_max = []
costs_max = []   

lengths_min = []
costs_min = []     

lengths_avg = []
costs_avg = []   

for idx, lengths in enumerate(global_lengths):
    if len(lengths) > 1:
        lengths_max.append(lengths[-1])
        costs_max.append(global_costs[idx][-1])
        lengths_min.append(lengths[0])
        costs_min.append(global_costs[idx][0])
    lengths_avg.append(np.array(lengths).mean())    
    costs_avg.append(np.array(global_costs[idx]).mean())   

print('AVG max len', round(np.asarray(max_lengths).mean(), 2),  round(np.asarray(max_lengths).std(), 2))
print('AVG max len costs', round(np.asarray(max_lengths_cost).mean(), 2),  round(np.asarray(max_lengths_cost).std(), 2))
print('-'*89)

print('AVG min cost', round(np.asarray(min_costs).mean(), 2),  round(np.asarray(min_costs).std(), 2))
print('AVG min cost len', round(np.asarray(min_costs_length).mean(), 2),  round(np.asarray(min_costs_length).std(), 2))
print('-'*89)


print('AVG len', round(np.asarray(lengths_avg).mean(), 2),  round(np.asarray(lengths_avg).std(), 2))
print('AVG costs', round(np.asarray(costs_avg).mean(), 2),  round(np.asarray(costs_avg).std(), 2))
print('-'*89)
print('AVG erdos ', round(np.asarray(max_erdos_numbers).mean(), 2), round(np.asarray(max_erdos_numbers).std(), 2))

