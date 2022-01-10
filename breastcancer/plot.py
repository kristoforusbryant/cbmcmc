#!/usr/bin/env python

import numpy as np
from utils.diagnostics import IAT
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

N, N_OBS = 93, 250
GNAME = 'breastcancer'

def plot_traces(edge_sizes, triangles_sizes):
    """ Plots traces of the Edges and Star cycles bases MCMC algorithms """
    fig, ax = plt.subplots()
    ax.plot(edge_sizes, label='Edges')
    ax.plot(triangles_sizes, label='Star Cycle Bases')

    ax.set_xlabel('Iterations', fontsize=15)
    ax.set_ylabel('Size', fontsize=15)
    ax.legend(loc='upper right', fontsize=12)

    y_pos = min(edge_sizes[1000:])
    txt = f"IAT: {round(IAT(edge_sizes), 3)} (Edges),\n"
    txt += f"IAT: {round(IAT(triangles_sizes), 3)} (Star Cycle Bases)"
    ax.text(1, y_pos, txt, fontsize=14)

    ax.set_title('Graph Sizes Trace Plot', fontsize=16)
    return fig, ax

def _get_adjm_split(adjm):
    triu = np.triu_indices(adjm.shape[0], 1)
    edgeM = adjm.copy()
    edgeM[triu] = edgeM[triu] > .5
    return edgeM

def plot_comparison(edge_adjm, triangles_adjm):
    """ Plots a figure comparing posterior edge inclusion probabilities of Edges and Star cycles bases algorithms """
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    for idx in range(2):
        im = axs[idx].pcolormesh(_get_adjm_split(
            [edge_adjm, triangles_adjm][idx]
        ).T, cmap="Greys", vmin=0.0, vmax=1.0)

        axs[idx].set_title(['Edge basis', 'Star cycle basis'][idx])
        axs[idx].tick_params(axis='both', which='major')
        axs[idx].axis("image")  # Ensure the plots are square.
        axs[idx].invert_yaxis()
        axs[idx].xaxis.set_label_position("top")
        axs[idx].set_xlabel("Vertex index")
        axs[idx].set_ylabel("Vertex index")

        axs[idx].tick_params(
            top=True, bottom=False, labeltop=True, labelbottom=False
        )

        labels = np.arange(
            start=10, stop=edge_adjm.shape[0] + 1, step=10, dtype=int
        )

        axs[idx].set_xticks(labels - 0.5)
        axs[idx].set_yticks(labels - 0.5)
        axs[idx].set_xticklabels(labels)
        axs[idx].set_yticklabels(labels)

    bar = fig.colorbar(
        im, ax=axs, shrink=0.5, orientation="horizontal", pad=0.05
    )
    
    bar.set_label(r"Posterior edge inclusion probability")
    return fig, axs

def plot_all_comparisons(edge_adjm, triangles_adjm, data):
    """ Plots a figure comparing covariance selection outcomes of the Edge and Star cycle bases algorithms with GraphicalLasso, GeneNet and BDgraph """
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
    return fig, axs


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
    fig, _ = plot_traces(edge_sizes, triangles_sizes)
    fig.savefig(f'figs/compare_traces_{GNAME}_{N}_{N_OBS}.pdf', bbox_inches='tight')

    print('Plotting Comparison between Edges and Star cycles bases')
    fig, _ = plot_comparison(edge_adjm, triangles_adjm)
    fig.savefig(f'figs/compare_edge_star_cycle_{GNAME}_{N}_{N_OBS}.pdf', bbox_inches='tight')

    print('Plotting Comparisons with glasso, BDgraph, and GeneNet')
    fig, _ = plot_all_comparisons(edge_adjm, triangles_adjm, data)
    fig.savefig(f'figs/compare_posterior_{GNAME}_{N}_{N_OBS}.pdf', bbox_inches='tight')

if __name__ == '__main__':
    main()
