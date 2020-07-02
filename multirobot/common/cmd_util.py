import argparse

import tensorflow as tf
from baselines.common import tf_util
from baselines.common.vec_env.vec_normalize import VecNormalize
from glog import info

from multirobot.common.vec_env.marl_vec_env import MarlVecEnv
from multirobot.environment.environment import MultiAgentEnv


def parse_args():
    """
    common arg parser
    @return:
    """
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=400, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=10000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=2048, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--load-dir", type=str, default="",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")

    # baselines ddpg

    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--env_type',
                        help='type of environment, used when the environment type cannot be automatically determined',
                        type=str)
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--alg', help='Algorithm', type=str, default='bl_maddpg')
    parser.add_argument('--num_timesteps', type=float, default=1e6),
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
    parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--num_env',
                        help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco',
                        default=None, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--result_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
    parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
    parser.add_argument('--play', default=False, action='store_true')

    parser.add_argument("--debug-display", action="store_true", default=False)
    parser.add_argument('--config_path', help='yaml to load env settings.', default=None, type=str)
    parser.add_argument('--log_path', help='yaml to load env settings.', default=None, type=str)
    parser.add_argument('--nb_epoch_cycles', type=int, default=3),
    parser.add_argument('--nb_rollout_steps', type=int, default=400),
    parser.add_argument('--nb_epochs', type=int, default=None),
    parser.add_argument("--save_rate", type=int, default=1,
                        help="save model once every time this many epochs are completed")
    parser.add_argument("--save_model", action="store_true", default=True)
    parser.add_argument("--save_actions", action="store_true", default=True)
    parser.add_argument("--save_video", action="store_true", default=False)

    return parser.parse_args()


def make_base_env(scenario_name, arglist, benchmark=False):
    """
    make a base environment multirobot.environment, from a multirobot.scenario
    @param scenario_name: scenario py file
    @param arglist: arglist
    @param benchmark: bool if enable benchmark
    @return: multirobot.environment
    """
    import multirobot.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    if arglist.config_path is not None:
        info('loading world config from ' + arglist.config_path)
        scenario.load(arglist.config_path, world)
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data,
                            scenario.done, True, reset_vehicle_callback=scenario.reset_vehicles)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            done_callback=scenario.done, shared_viewer=True,
                            reset_vehicle_callback=scenario.reset_vehicles)
        # env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
        #                     done_callback=scenario.done, shared_viewer=True)
    return env


def build_env(arglist):
    """
    build wrapped env according to algorithms chosen in arglist
    @param arglist: arglist
    @return: env
    """
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    tf_util.get_session(config=config)

    base_env = make_base_env(arglist.scenario, arglist, arglist.benchmark)

    if arglist.alg == 'bl_maddpg':
        from multirobot.maddpg.environment import BlMaddpgEnv
        env = BlMaddpgEnv(base_env)
    else:
        raise NotImplementedError

    env = MarlVecEnv(env)
    # env = VecNormalize(env, use_tf=False)  # not use tf to avoid name conflict
    return env
