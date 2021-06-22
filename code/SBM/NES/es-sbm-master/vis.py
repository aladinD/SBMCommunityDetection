import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def plot_soft_community_graph(graph: nx.Graph, assignments: np.array) -> None:
    colors = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9']
    pos = nx.spring_layout(graph)
    ax = plt.subplot()
    nx.draw_networkx_edges(graph, pos=pos, ax=ax)
    for n in graph.nodes():
        ax.pie(assignments[n, :], center=pos[n], radius=0.05, colors=colors[:assignments.shape[1]])
    plt.tight_layout()
    #plt.savefig('./img/soft-com.svg')
    plt.close()