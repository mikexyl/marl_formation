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
    def __init__(self, arglist):
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

        self.actions = None

    def configure_logger(self, log_path, **kwargs):
        if log_path is not None:
            logger.configure(log_path)
        else:
            logger.configure(**kwargs)

    def save_model(self):
        if self.model_file is not None:
            info('saving vars to ' + self.model_file)
            U.save_variables(self.model_file)
        else:
            info('save_path is None, not saving')

    def load_model(self):
        if self.model_file is not None \
                and os.path.exists(self.model_file):
            info('loading vars from ' + self.model_file)
            U.load_variables(self.model_file)
        else:
            info('load_path is none or file not exists, not loading')

    def add_action(self, action):
        if self.actions is None:
            self.actions = action.flatten()
        else:
            self.actions = np.vstack((self.actions, action.flatten()))

    def save_actions(self, epoch, cycle):
        actions_file = os.path.join(self.actions_path, '%04d-%04d.csv' % (epoch, cycle))
        np.savetxt(actions_file, self.actions, delimiter=',')
        self.actions = None

    def get_paths(self):
        return self.result_path, self.log_path, self.model_file, self.actions_path