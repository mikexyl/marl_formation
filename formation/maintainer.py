from formation.graph import Graph


class Maintainer(object):
    def __init__(self):
        self.t_formation = None
        self.c_formation = None
        self.n = None

    def set_num_vehicles(self, num_vehicles):
        self.t_formation = Graph(num_vehicles, directed=False)
        self.c_formation = Graph(num_vehicles, directed=False)
        self.n = num_vehicles

    def assign(self):
        raise NotImplementedError

    def load_target_formation(self, config):
        raise NotImplementedError

    def add_edges(self, es):
        self.c_formation.add_edges(es, weights=True)

    def reset(self):
        self.c_formation.delete_edges(None)

    def formation_exam(self):
        return 0, False