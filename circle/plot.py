#!/usr/bin/env python

import sys, pickle
import numpy as np 

from cbmcmc import Uniform, StarCycleBases, OneEdge, MCMC_Sampler_Ratio, GW_Ratio
from utils.myGraph import Graph
from utils.diagnostics import IAT, str_list_to_adjm

def main():
    n, n_obs = 30, 20
    gname = 'circle'

    with open(f"data/graph_{gname}_{n}.pkl", 'rb') as handle:
        g = pickle.load(handle)

    K = np.loadtxt(f"data/{gname}_precision_{n}.csv", delimiter=",")
    data = np.loadtxt(f"data/{gname}_data_{n}_{n_obs}.csv", delimiter=",")

    # Edges
    edge_sizes = np.loadtxt(f"res/edge_sizes_{gname}_{n}_{n_obs}.csv", delimiter=',')
    edge_adjm = np.loadtxt(f"res/edge_adjm_{gname}_{n}_{n_obs}.csv", delimiter=',')

    # StarCycleBases
    triangles_sizes = np.loadtxt(f"res/triangles_sizes_{gname}_{n}_{n_obs}.csv", delimiter=',')
    triangles_adjm = np.loadtxt(f"res/triangles_adjm_{gname}_{n}_{n_obs}.csv", delimiter=',')

    # Plotting Trace Plot
    import matplotlib.pyplot as plt
    plt.plot(edge_sizes, label='Edges')
    plt.plot(triangles_sizes, label='Star Cycle Bases')

    plt.xlabel('Iterations', fontsize=15)
    plt.ylabel('Size', fontsize=15)
    plt.legend(loc='upper right', fontsize=12)

    y_pos = min(edge_sizes[1000:])
    txt = f"IAT: {round(IAT(edge_sizes), 3)} (Edges),\n"
    txt += f"IAT: {round(IAT(triangles_sizes), 3)} (Star Cycle Bases)"
    plt.text(1, y_pos, txt, fontsize=14)

    plt.title('Graph Sizes Trace Plot', fontsize=16)
    plt.savefig(f'figs/compare_{gname}_{n}_{n_obs}_traces.pdf', bbox_inches='tight')

    # Plotting Comparisons with glasso, BDgraph, and GeneNet
    from sklearn.covariance import GraphicalLassoCV
    glasso = GraphicalLassoCV(max_iter=1000).fit(data)
    glasso_prec = np.abs(glasso.precision_)
    glasso_adjm = glasso_prec - np.diag(np.diag(glasso_prec))

    bd_adjm = np.loadtxt(f'res/bd_adjm_{gname}_{n}_{n_obs}.csv', delimiter=' ')
    bd_adjm += np.transpose(bd_adjm)

    gnet_adjm1 = np.loadtxt(f'res/gene_net_pval_1_{gname}_{n}_{n_obs}.csv', delimiter=' ')
    gnet_adjm2 = np.loadtxt(f'res/gene_net_pval_2_{gname}_{n}_{n_obs}.csv', delimiter=' ')
    gnet_adjm3 = np.loadtxt(f'res/gene_net_pval_3_{gname}_{n}_{n_obs}.csv', delimiter=' ')

    def get_adjm_split(adjm):
        tril = np.tril_indices(adjm.shape[0], 1)
        edgeM = adjm.copy()
        edgeM[tril] = edgeM[tril] > .5
        return edgeM

    fig, axs = plt.subplots(2, 5, figsize=(25, 10))

    axs[0, 0].imshow(g.GetAdjM())
    axs[0, 0].set_title("True Graph", fontsize=20)
    K_ = K.copy()
    K_[np.eye(30, dtype=bool)] = 0
    axs[0, 1].imshow(np.abs(K_))
    axs[0, 1].set_title("True Precision Matrix", fontsize=20)
    emp_K = np.linalg.inv(data.transpose() @ data)
    axs[0, 2].imshow(emp_K)
    axs[0, 2].set_title("Empirical Precision Matrix", fontsize=20)
    axs[0, 3].imshow(get_adjm_split(edge_adjm))
    axs[0, 3].set_title("Edges", fontsize=20)
    axs[0, 4].imshow(get_adjm_split(triangles_adjm))
    axs[0, 4].set_title("Star Cycle Bases", fontsize=20)
    axs[1, 0].imshow(glasso_adjm)
    axs[1, 0].set_title("Glasso", fontsize=20)
    axs[1, 1].imshow(get_adjm_split(bd_adjm))
    axs[1, 1].set_title("BDgraph", fontsize=20)
    axs[1, 2].imshow(get_adjm_split(gnet_adjm1))
    axs[1, 2].set_title("GeneNet (pval < 0.001)", fontsize=20)
    axs[1, 3].imshow(get_adjm_split(gnet_adjm2))
    axs[1, 3].set_title("GeneNet (pval < 0.01)", fontsize=20)
    axs[1, 4].imshow(get_adjm_split(gnet_adjm3))
    axs[1, 4].set_title("GeneNet (pval < 0.05)", fontsize=20)

    fig.suptitle('Comparing Posterior Inference against other Covariance Selection Algorithms', fontsize=30)
    fig.savefig(f'figs/comparison_{gname}_{n}_{n_obs}.pdf', bbox_inches='tight')

if __name__ == '__main__':
    main()
