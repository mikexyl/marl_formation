from baselines import logger

from multirobot.maddpg.environment import BaseLinesEnv
from multirobot.environment.environment import make_env
from multirobot.common.util import parse_args

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def train(arglist):
    # Create environment
    env = make_env(arglist.scenario, arglist, arglist.benchmark)
    ddpg_env = BaseLinesEnv(env)

    from multirobot.common.saver import Saver
    saver=Saver(arglist)

    from multirobot.maddpg.maddpg import learn
    learn(network="mlp",
          env=ddpg_env,
          # total_timesteps=400,
          nb_epochs=arglist.nb_epochs,
          nb_epoch_cycles=arglist.nb_epoch_cycles,
          nb_rollout_steps=arglist.nb_rollout_steps,
          render=arglist.display,
          normalize_observations=False,
          normalize_returns=False,
          save_model=arglist.save_model,
          save_rate=arglist.save_rate,
          restore=arglist.restore,
          save_actions=arglist.save_actions,
          saver=saver
          )


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
