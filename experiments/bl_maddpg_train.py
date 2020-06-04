from baselines import logger

from multirobot.common.cmd_util import build_env
from multirobot.common.cmd_util import parse_args
from multirobot.common.saver import Saver
from multirobot.common.vec_env.vec_video_recorder import VecVideoRecorder

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
    # Create saver
    # TODO saver not very professional yet
    saver = Saver(arglist)

    # Create environment
    env = build_env(arglist)
    if arglist.save_video:
        env = VecVideoRecorder(env, saver.videos_path,
                               record_video_trigger=lambda cycle: cycle == 0)

    if arglist.alg == 'bl_maddpg':
        from multirobot.maddpg.maddpg import learn
        learn(network="mlp",
              env=env,
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
    else:
        raise NotImplementedError


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
