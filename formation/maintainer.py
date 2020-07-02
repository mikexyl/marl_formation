from math import sqrt

from formation.graph import Graph
import networkx as nx
import numpy as np


class Maintainer(object):
    """
    Formation maintainer to monitor the formation, and check if the formation is formed, and compute the error.
    It maintain a graph of current formation called self.c_formation, and a target formation self.t_formation
    self.n:
    @var self.t_formation: target formation
    @var self.c_formation: current formation
    @var self.n: vehicle number
    @var self.updated: if c_formation is updated in the current episode, to avoid repeated update
    """
    def __init__(self):
        self.t_formation = None
        self.c_formation = None
        self.n = None
        self.updated = False

    def set_num_vehicles(self, num_vehicles):
        """
        set the number of vehicles in the formation
        @param num_vehicles: number of vehicles
        @return: no return
        """
        self.t_formation = Graph(num_vehicles)
        self.c_formation = Graph(num_vehicles)
        self.n = num_vehicles

    def assign(self):
        """
        assign vehicles to formation positions actively
        @return:
        """
        raise NotImplementedError

    def load_target_formation(self, config):
        """
        Load predefined target formation
        @param config: target formation configuration
        @return:
        """
        raise NotImplementedError

    def load_sample_formation(self, formation='regular_polygon', edge_length=1):
        """
        Set sample regular polygon formation, currently 3 or 4 vehicles
        @param formation: formation type, only 'regular_polygon' is supported now
        @param edge_length: length of regular polygon
        @return:
        """
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
        """
        add edges to current formation graph
        @param es: list of edges, edge[0:1]: id of nodes of connection, edge[2] is the distance
        @return:
        """
        for edge in es:
            # info("added: %d, %d" % (edge[0], edge[1]))
            self.c_formation.add_edge(edge[0], edge[1], weight=edge[2])

    def reset(self):
        """
        delete all edges, and set self.updated to False
        @return:
        """
        self.c_formation.delete_edges()
        self.updated = False

    # todo consider stability
    def formation_exam(self, eps_form):
        """
        exam if the required formation is reached, and return the error
        @param eps_form: threshold value of formation error
        @return: 0, 0: no enough edges to form minimal graph
                disp, 2: if threshold is reached
                disp, 1: if threshold not reached
        """
        # info("current graph size: %d" % self.c_formation.size())
        if self.c_formation.size() < (2 * self.n - 3):
            return 0, 0
        else:
            # todo this is a sample logic for triangular shape
            if self.n == 3:
                sorted_es_c = self.c_formation.sorted_edges
                sorted_es_t = self.t_formation.sorted_edges
                disp = abs(sorted_es_c[:, 2] - sorted_es_t[:, 2])
                return disp, 2 if all(disp < eps_form) else 1
            else:
                raise NotImplementedError

    @property
    def c_adjencency_matrix(self):
        """

        @return: adjecency matrix of current formation graph
        """
        return np.array(nx.adjacency_matrix(self.c_formation).todense())
