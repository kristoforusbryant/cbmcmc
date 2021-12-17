#!/usr/bin/env python

import numpy as np
from utils.diagnostics import IAT
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def main():
    n, n_obs = 93, 250
    gname = 'breastcancer'
    data = np.loadtxt(f"data/{gname}_data_{n}_{n_obs}.csv", delimiter=",")

    print('Loading results...')
    # Edges
    edge_sizes = np.loadtxt(f"res/edge_sizes_{gname}_{n}_{n_obs}.csv", delimiter=',')
    edge_adjm = np.loadtxt(f"res/edge_adjm_{gname}_{n}_{n_obs}.csv", delimiter=',')

    # StarCycleBases
    triangles_sizes = np.loadtxt(f"res/triangles_sizes_{gname}_{n}_{n_obs}.csv", delimiter=',')
    triangles_adjm = np.loadtxt(f"res/triangles_adjm_{gname}_{n}_{n_obs}.csv", delimiter=',')


    print('Plotting Trace Plots...')
    # plt.plot(edge_sizes, label='Edges')
    # plt.plot(triangles_sizes, label='Star Cycle Bases')

    # plt.xlabel('Iterations', fontsize=15)
    # plt.ylabel('Size', fontsize=15)
    # plt.legend(loc='upper right', fontsize=12)

    # y_pos = min(edge_sizes[1000:])
    # txt = f"IAT: {round(IAT(edge_sizes), 3)} (Edges),\n"
    # txt += f"IAT: {round(IAT(triangles_sizes), 3)} (Star Cycle Bases)"
    # plt.text(1, y_pos, txt, fontsize=14)

    # plt.title('Graph Sizes Trace Plot', fontsize=16)
    # plt.savefig(f'figs/compare_traces_{gname}_{n}_{n_obs}.pdf', bbox_inches='tight')

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
    gnet_adjm4 = np.loadtxt(f'res/gene_net_pval_4_{gname}_{n}_{n_obs}.csv', delimiter=' ')

    def get_adjm_split(adjm):
        tril = np.tril_indices(adjm.shape[0], 1)
        edgeM = adjm.copy()
        edgeM[tril] = edgeM[tril] > .5
        return edgeM

    print('Plotting Heatmaps...')
    fig, axs = plt.subplots(2, 4, figsize=(40, 20))

    titles = ["Edge", "Star Cycle Bases", "BDgraph", "Graphical Lasso", "GeneNet (pval < 0.001)", "GeneNet (pval < 0.01)", "GeneNet (pval < 0.05)", "GeneNet (pval < 0.1)"]
    adjms = [get_adjm_split(edge_adjm), get_adjm_split(triangles_adjm), get_adjm_split(bd_adjm), glasso_adjm,\
                 get_adjm_split(gnet_adjm1), get_adjm_split(gnet_adjm2), get_adjm_split(gnet_adjm3), get_adjm_split(gnet_adjm4)]

    for i in range(2):
        for j in range(4):
            idx = i * 4 + j
            if i == 0:
                divider = make_axes_locatable(axs[i, j])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                im = axs[i, j].imshow(adjms[idx], cmap='Greys')
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=18)
            else:
                axs[i, j].imshow(adjms[idx], cmap='Greys')
            axs[i, j].set_title(titles[idx], fontsize=42)
            axs[i, j].tick_params(axis='both', which='major', labelsize=22)

    fig.suptitle('Comparing against other Covariance Selection Algorithms', fontsize=55)
    fig.savefig(f'figs/compare_posterior_{gname}_{n}_{n_obs}.pdf', bbox_inches='tight')

if __name__ == '__main__':
    main()
