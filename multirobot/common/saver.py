import os

import baselines.common.tf_util as U
import numpy as np
from baselines import logger
from glog import info

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class Saver(object):
    """
    Saver class that saves data like network params, actions, etc, and generates saving path for other savings happening
    outside this class
    """
    def __init__(self, arglist):
        """
        init saver paths and make relative directories
        @param arglist: create save paths and files inside arglist.result_path
        """
        if arglist.result_path is None:
            import subprocess
            p = subprocess.Popen('git rev-parse HEAD', shell=True, stdout=subprocess.PIPE)
            out, err = p.communicate()
            self.result_path = os.path.join('/tmp', 'results', str(out.splitlines()[0]).split('\'')[1])
        else:
            self.result_path = os.path.expanduser(arglist.result_path)

        self.log_path = os.path.join(self.result_path, 'log')
        os.makedirs(self.log_path, exist_ok=True)

        if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
            rank = 0
            self.configure_logger(self.log_path)
        else:
            rank = MPI.COMM_WORLD.Get_rank()
            self.configure_logger(self.log_path, format_strs=[])

        self.model_path = os.path.join(self.result_path, 'model')
        self.model_file = os.path.join(self.model_path, 'checkpoint')

        self.actions_path = os.path.join(self.result_path, 'actions')
        os.makedirs(self.actions_path, exist_ok=True)
        self.plots_path = os.path.join(self.result_path, 'plots')
        os.makedirs(self.plots_path, exist_ok=True)
        self.videos_path = os.path.join(self.result_path, 'videos')
        os.makedirs(self.videos_path, exist_ok=True)

        self.actions = None

    def configure_logger(self, log_path, **kwargs):
        """
        config baselines.logger
        @param log_path: log_path to save logs
        @param kwargs: other params
        @return:
        """
        if log_path is not None:
            logger.configure(log_path)
        else:
            logger.configure(**kwargs)

    def save_model(self):
        """
        save model variables using tensorflow's function saving all vars in a session
        @return:
        """
        if self.model_file is not None:
            info('saving vars to ' + self.model_file)
            U.save_variables(self.model_file)
        else:
            info('save_path is None, not saving')

    def load_model(self):
        """
        load vars from the saved model file under saving path
        @return:
        """
        if self.model_file is not None \
                and os.path.exists(self.model_file):
            info('loading vars from ' + self.model_file)
            U.load_variables(self.model_file)
        else:
            info('load_path is none or file not exists, not loading')

    def add_action(self, action):
        """
        cache a action
        @param action: action to cache
        @return:
        """
        if self.actions is None:
            self.actions = action.flatten()
        else:
            self.actions = np.vstack((self.actions, action.flatten()))

    def save_actions(self, epoch, cycle):
        """
        save all cached actions
        @param epoch: epoch these actions belongs
        @param cycle: cycle these actions belongs
        @return:
        """
        actions_file = os.path.join(self.actions_path, '%04d-%04d.csv' % (epoch, cycle))
        np.savetxt(actions_file, self.actions, delimiter=',')
        self.actions = None

    def get_paths(self):
        """
        get all paths
        @return: result_path, log_path, model_file, actions_path, videos_path
        """
        return self.result_path, self.log_path, self.model_file, self.actions_path, self.videos_path
