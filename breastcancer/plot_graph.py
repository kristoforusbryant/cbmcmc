#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from utils.myGraph import Graph

THRESH = [0.5, 0.8, 0.9, 0.95]
CIRCLE = False

n, n_obs = 93, 250
gname = 'breastcancer'

edge_adjm = np.loadtxt(f"res/edge_adjm_{gname}_{n}_{n_obs}.csv", delimiter=',')
triangles_adjm = np.loadtxt(f"res/triangles_adjm_{gname}_{n}_{n_obs}.csv", delimiter=',')

def draw_graph(thresh=.5):
    """ Draws the p-percentile graphs obtained from the Edge and Star cycle bases algorithms """
    # Edges
    edge_g = Graph(93)
    edge_g.SetFromAdjM(np.array(edge_adjm > thresh, dtype=int))

    # StarCycleBases
    star_g = Graph(93)
    star_g.SetFromAdjM(np.array(triangles_adjm > thresh, dtype=int))

    ## Getting Position
    if CIRCLE:
        edol = { k:v for k,v in edge_g._dol.items()}
        sdol = { k:v for k,v in star_g._dol.items()}
        pos = edge_g.GetCirclePos()
    else:
        edol = { k:v for k,v in edge_g._dol.items() if len(v) > 0 }
        sdol = { k:v for k,v in star_g._dol.items() if len(v) > 0 }
        if bool(set(edol) - set(sdol)) & bool(not (set(sdol) - set(edol))):
            # edol is the bigger set
            G = nx.from_dict_of_lists(edol)
            pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=LR')
        elif bool(set(sdol) - set(edol)) & (not (set(edol) - set(sdol))):
            # sdol is the bigger set
            G = nx.from_dict_of_lists(sdol)
            pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=LR')
        else:
            for k in sdol:
                if k not in edol:
                    edol[k] = []

            G = nx.from_dict_of_lists(edol)
            pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=LR')

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    labels = {k:k + 1 for k in edol}
    nx_edge_g = nx.from_dict_of_lists(edol)
    nx.draw_networkx_edges(nx_edge_g, pos=pos, ax=axs[0], width=2., alpha=.3)
    nx.draw_networkx_nodes(nx_edge_g, pos=pos, ax=axs[0], node_size=200, label=np.arange(1,94))
    nx.draw_networkx_labels(nx_edge_g, pos=pos, ax=axs[0], labels=labels, font_size=12)


    labels = {k:k + 1 for k in sdol}
    nx_star_g = nx.from_dict_of_lists(sdol)
    nx.draw_networkx_edges(nx_star_g, pos=pos, ax=axs[1], width=2., alpha=.3)
    nx.draw_networkx_nodes(nx_star_g, pos=pos, ax=axs[1], node_size=200, label=np.arange(1, 94))
    nx.draw_networkx_labels(nx_star_g, pos=pos, ax=axs[1], labels=labels, font_size=12)

    axs[0].set_title('Edge', fontsize=30)
    axs[1].set_title('Star Cycle', fontsize=30)

    axs[0].set_frame_on(False)
    axs[1].set_frame_on(False)

    return fig, axs


if __name__ == '__main__':
    for thresh in THRESH:
        if CIRCLE:
            fig, axs = draw_graph(thresh)
            fig.savefig(f"figs/graph_thresh-{thresh}-circle.pdf", bbox_inches='tight')
        else:
            fig, axs = draw_graph(thresh)
            fig.savefig(f"figs/graph_thresh-{thresh}.pdf", bbox_inches='tight')
