#!/usr/bin/env python3
from cbmcmc import Uniform, StarCycleBases, OneEdge, MCMC_Sampler, GW_Ratio
from sklearn.covariance import GraphicalLassoCV
import numpy as np
import pickle
from utils.myGraph import Graph
from multiprocessing import Pool
from itertools import product

n = 93
prop_l = [OneEdge, StarCycleBases]
prop_lab = ['edge', 'star-cycle-bases']

rep_l = np.arange(1)
n_obs_l = [250]

iter = 1000000
burnin = 100000
true_graph = 'breastcancer'

def run(l):
    np.random.seed(123)
    prop_idx, rep, n_obs = l

    prior = Uniform(n, Param)
    prop = prop_l[prop_idx](n, Param)

    data = np.loadtxt('data/data.csv', delimiter=',')
    delta = 3
    D = np.eye(n)
    lik = GW_Ratio(data, delta, D, Param)

    # Initialising from Graphical Lasso estimate
    K = GraphicalLassoCV(max_iter=100000).fit(data).get_precision()
    adjm = np.array(np.abs(K) > .5, dtype=int)
    g = Graph(93)
    g.SetFromAdjM(adjm)

    sampler = MCMC_Sampler_Ratio(prior, prop, lik, data, outfile=f"res/sampler_{prop_lab[prop_idx]}_{n}_{true_graph}-{rep}_{n_obs}.pkl")
    sampler.run(iter, fixed_init=g)
    sampler.save_object()
    summ = sampler.get_summary(g, b=burnin, inc_distances=False, acc_scaled_size=1000)
    with open(f'res/res/sampler_{prop_lab[prop_idx]}_{n}_{true_graph}-{rep}_{n_obs}_summary.pkl') as handle:
        pickle.dump(summ, handle)

    
pool = Pool()
pool.map(run, list(product([0, 1], rep_l, n_obs_l))) # prop x rep x n_obs x pcor
