#!/usr/bin/env python

import numpy as np
from utils.diagnostics import IAT
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

N, N_OBS = 93, 250
GNAME = 'breastcancer'

def plot_traces(edge_sizes, triangles_sizes):
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
    plt.savefig(f'figs/compare_traces_{GNAME}_{N}_{N_OBS}.pdf', bbox_inches='tight')

def _get_adjm_split(adjm):
    triu = np.triu_indices(adjm.shape[0], 1)
    edgeM = adjm.copy()
    edgeM[triu] = edgeM[triu] > .5
    return edgeM

def plot_comparison(edge_adjm, triangles_adjm):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = axs[0].pcolormesh(np.rot90(_get_adjm_split(edge_adjm)), cmap="Greys", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=18)
    axs[0].set_title('Edge', fontsize=42)
    axs[0].tick_params(axis='both', which='major', labelsize=22)

    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = axs[1].pcolormesh(np.rot90(_get_adjm_split(triangles_adjm)), cmap="Greys", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=18)
    axs[1].set_title('Star Cycle Bases', fontsize=42)
    axs[1].tick_params(axis='both', which='major', labelsize=22)

    fig.suptitle('Comparing Edge and Star Cycle Bases', fontsize=45, y=1.05)
    fig.savefig(f'figs/compare_edge_star_cycle_{GNAME}_{N}_{N_OBS}.pdf', bbox_inches='tight')

def plot_all_comparisons(edge_adjm, triangles_adjm, data):
    from sklearn.covariance import GraphicalLassoCV
    glasso = GraphicalLassoCV(max_iter=1000).fit(data)
    glasso_prec = np.abs(glasso.precision_)
    glasso_adjm = glasso_prec - np.diag(np.diag(glasso_prec))

    bd_adjm = np.loadtxt(f'res/bd_adjm_{GNAME}_{N}_{N_OBS}.csv', delimiter=' ')
    bd_adjm += np.transpose(bd_adjm)

    gnet_adjm1 = np.loadtxt(f'res/gene_net_pval_1_{GNAME}_{N}_{N_OBS}.csv', delimiter=' ')
    gnet_adjm2 = np.loadtxt(f'res/gene_net_pval_2_{GNAME}_{N}_{N_OBS}.csv', delimiter=' ')
    gnet_adjm3 = np.loadtxt(f'res/gene_net_pval_3_{GNAME}_{N}_{N_OBS}.csv', delimiter=' ')
    gnet_adjm4 = np.loadtxt(f'res/gene_net_pval_4_{GNAME}_{N}_{N_OBS}.csv', delimiter=' ')

    print('Plotting Heatmaps...')
    fig, axs = plt.subplots(2, 4, figsize=(40, 20))

    titles = ["Edge", "Star Cycle Bases", "BDgraph", "Graphical Lasso", "GeneNet (pval < 0.001)", "GeneNet (pval < 0.01)", "GeneNet (pval < 0.05)", "GeneNet (pval < 0.1)"]
    adjms = [_get_adjm_split(edge_adjm), _get_adjm_split(triangles_adjm), _get_adjm_split(bd_adjm), glasso_adjm,\
                 _get_adjm_split(gnet_adjm1), _get_adjm_split(gnet_adjm2), _get_adjm_split(gnet_adjm3), _get_adjm_split(gnet_adjm4)]

    for i in range(2):
        for j in range(4):
            idx = i * 4 + j
            if i == 0:
                divider = make_axes_locatable(axs[i, j])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                im = axs[i, j].pcolormesh(np.rot90(adjms[idx]), cmap="Greys", vmin=0.0, vmax=1.0)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=18)
            else:
                axs[i, j].imshow(adjms[idx], cmap='Greys')
            axs[i, j].set_title(titles[idx], fontsize=42)
            axs[i, j].tick_params(axis='both', which='major', labelsize=22)

    fig.suptitle('Comparing against other Covariance Selection Algorithms', fontsize=55)
    fig.savefig(f'figs/compare_posterior_{GNAME}_{N}_{N_OBS}.pdf', bbox_inches='tight')


def main():
    data = np.loadtxt(f"data/{GNAME}_data_{N}_{N_OBS}.csv", delimiter=",")

    print('Loading results...')
    # Edges
    edge_sizes = np.loadtxt(f"res/edge_sizes_{GNAME}_{N}_{N_OBS}.csv", delimiter=',')
    edge_adjm = np.loadtxt(f"res/edge_adjm_{GNAME}_{N}_{N_OBS}.csv", delimiter=',')

    # StarCycleBases
    triangles_sizes = np.loadtxt(f"res/triangles_sizes_{GNAME}_{N}_{N_OBS}.csv", delimiter=',')
    triangles_adjm = np.loadtxt(f"res/triangles_adjm_{GNAME}_{N}_{N_OBS}.csv", delimiter=',')

    print('Plotting Trace Plots...')
    plot_traces(edge_sizes, triangles_sizes)

    print('Plotting Comparison between Edges and Star cycles bases')
    plot_comparison(edge_adjm, triangles_adjm)

    print('Plotting Comparisons with glasso, BDgraph, and GeneNet')
    plot_all_comparisons(edge_adjm, triangles_adjm, data)

if __name__ == '__main__':
    main()
