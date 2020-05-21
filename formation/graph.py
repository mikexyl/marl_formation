import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def cmp_edges(e1, e2):
    label = 'weight'
    if e1[2] < e2[2]:
        return -1
    elif e1[2] == e2[2]:
        return 0
    else:
        return -1


class Graph(nx.Graph):
    def __init__(self, num_v, directed=False):
        super(Graph, self).__init__()
        self.add_nodes_from(range(0, num_v))
        self.id_map = None
        self.n = num_v
        self.sorted_es = None

    def load(self, config):
        raise NotImplementedError

    def draw_graph(self):
        pos = nx.spring_layout(self)

        # nodes
        nx.draw_networkx_nodes(self, pos, node_size=700)
        # edges
        nx.draw_networkx_edges(self, pos, edgelist=self.edges(data=True),
                               width=6)
        # labels
        labels = nx.get_edge_attributes(self, 'weight')
        nx.draw_networkx_edge_labels(self, pos, edge_labels=labels)
        nx.draw_networkx_labels(self, pos, font_size=20, font_family='sans-serif')

        plt.axis('off')
        plt.show()

    def delete_edges(self):
        self.clear()
        self.add_nodes_from(range(0, self.n))
        self.sorted_es = None

    @property
    def sorted_edges(self):
        if self.sorted_es is None:
            import functools
            self.sorted_es = np.array(sorted(self.edges.data('weight'), key=functools.cmp_to_key(cmp_edges)))
            return self.sorted_es
        else:
            return self.sorted_es
