from math import sqrt

from glog import info

from formation.graph import Graph


class Maintainer(object):
    def __init__(self):
        self.t_formation = None
        self.c_formation = None
        self.n = None

    def set_num_vehicles(self, num_vehicles):
        self.t_formation = Graph(num_vehicles)
        self.c_formation = Graph(num_vehicles)
        self.n = num_vehicles

    def assign(self):
        raise NotImplementedError

    def load_target_formation(self, config):
        raise NotImplementedError

    def load_sample_formation(self, formation='regular_polygon', edge_length=1):
        if formation == 'regular_polygon':
            if self.n > 4:
                print("currently, regular polygon shape only supports n=3 or 4")
                raise NotImplementedError
            for v1 in range(self.n):
                for v2 in range(v1 + 1, self.n):
                    if v1 is v2:
                        continue
                    if abs(v1 - v2) == 1 or self.n - abs(v1 - v2) == 1:
                        self.t_formation.add_edge(v1, v2, weight=edge_length)
                    else:
                        self.t_formation.add_edge(v1, v2, weight=edge_length * sqrt(2))
        else:
            raise NotImplementedError

    def add_edges(self, es):
        for edge in es:
            # info("added: %d, %d" % (edge[0], edge[1]))
            self.c_formation.add_edge(edge[0], edge[1], weight=edge[2])

    def reset(self):
        self.c_formation.delete_edges()

    # todo consider stability
    def formation_exam(self):
        # info("current graph size: %d" % self.c_formation.size())
        if self.c_formation.size() < (2 * self.n - 3):
            return 0, False
        else:
            return 0, False
