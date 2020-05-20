from igraph import Graph as igGraph


class Graph(igGraph):
    def __init__(self, num_v, directed=False):
        super(Graph, self).__init__(num_v, edges=None, directed=directed)
        self.id_map = None

    def load(self, config):
        raise NotImplementedError
