#!/usr/bin/env python

"""
A Python script that runs the Edge basis and Cycle basis MCMC algorithms.
"""
import sys, pickle
import numpy as np

from cbmcmc import Uniform, StarCycleBases, OneEdge, MCMC_Sampler_Ratio, GW_Ratio
from utils.myGraph import Graph
from utils.diagnostics import str_list_to_adjm


def main():
    n, n_obs = 30, 20
    gname = 'circle'
    
    # Edges
    data = np.loadtxt(f"data/{gname}_data_{n}_{n_obs}.csv", delimiter=",")

    prior = Uniform(n, Graph)
    prop = OneEdge(n, Graph)
    delta = 3
    D = np.eye(n)
    lik = GW_Ratio(data, delta, D, Graph)

    s = MCMC_Sampler_Ratio(prior, prop, lik, data)

    np.random.seed(123)
    s.run(50000, Graph(n))

    edge_sizes = [np.sum(np.array(list(st), dtype=int)) for st in s.res['SAMPLES'][5000::5]]
    edge_adjm = str_list_to_adjm(n, s.res['SAMPLES'][5000::5])

    np.savetxt(f"res/edge_sizes_{gname}_{n}_{n_obs}.csv", edge_sizes, delimiter=',')
    np.savetxt(f"res/edge_adjm_{gname}_{n}_{n_obs}.csv", edge_adjm, delimiter=',')

    del s

    # StarCycleBases
    prior = Uniform(n, Graph)
    prop = StarCycleBases(n, Graph)
    delta = 3
    D = np.eye(n)
    lik = GW_Ratio(data, delta, D, Graph)

    s_ = MCMC_Sampler_Ratio(prior, prop, lik, data)

    np.random.seed(123)
    s_.run(50000, Graph(n))

    triangle_sizes = [np.sum(np.array(list(st), dtype=int)) for st in s_.res['SAMPLES'][5000::5]]
    triangle_adjm = str_list_to_adjm(n, s_.res['SAMPLES'][5000::5])

    np.savetxt(f"res/triangles_sizes_{gname}_{n}_{n_obs}.csv", triangle_sizes, delimiter=',')
    np.savetxt(f"res/triangles_adjm_{gname}_{n}_{n_obs}.csv", triangle_adjm, delimiter=',')

    del s_

if __name__ == '__main__':
    main()
