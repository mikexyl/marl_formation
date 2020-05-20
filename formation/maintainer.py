from formation.graph import Graph


class Maintainer(object):
    def __init__(self, num_vehicles):
        self.t_formation = Graph(num_vehicles)
        self.c_formation = Graph(num_vehicles)
        self.n = num_vehicles

    def assign(self):
        raise NotImplementedError

    def load_target_formation(self, config):
        raise NotImplementedError

    def add_edges(self, es):
        raise NotImplementedError
