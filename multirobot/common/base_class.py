from abc import ABC, abstractmethod
from collections import OrderedDict


class BaseRLModel(ABC):
    def __init__(self):
        self.sess = None

    @abstractmethod
    def initialize(self, sess):
        pass

    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def get_parameter_list(self):
        pass

    def get_parameters(self):
        """
        Get current model parameters as dictionary of variable name -> ndarray.

        :return: (OrderedDict) Dictionary of variable name -> ndarray of model's parameters.
        """
        parameters = self.get_parameter_list()
        parameter_values = self.sess.run(parameters)
        return_dictionary = OrderedDict((param.name, value) for param, value in zip(parameters, parameter_values))
        return return_dictionary

    @abstractmethod
    def save(self, save_path, cloudpickle=False):
        raise NotImplementedError()

    @abstractmethod
    def load(self, load_path, **kwargs):
        raise NotImplementedError()

class BaseMultiAgentRLModel(ABC):
    def __init__(self, nb_agents):
        self.nb_agents=nb_agents
        self.agents=[None for _ in range(nb_agents)]

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def setup_model(self):
        pass

    def get_parameters(self):
        param_dict=OrderedDict()
        param_dict.update(agent.get_parameters() for agent in self.agents)
