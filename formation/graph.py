from igraph import Graph as igGraph


class Graph(igGraph):
    def __init__(self, num_v):
        super(Graph, self).__init__()
        super(Graph, self).add_vertices(num_v)
        self.id_map = None

    def load(self, config):
        raise NotImplementedError

