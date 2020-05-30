import argparse
import math

import numpy as np
from multiagent.core import Entity


def add_to_obs_grid(agent, entity, obs, label):
    observed = False
    entity_pos = entity.state.p_pos - agent.state.p_pos
    entity_polar = cart_to_polar(entity_pos)
    [i, j] = find_grid_id(agent, entity_polar)
    if i is not None and j is not None:
        obs[i, j] = label
        observed = True
        # glog.info([i, j, label])
    return obs, observed


# todo may need a util class for static methods
def cart_to_polar(cart):
    polar = np.zeros(2)
    polar[0] = np.linalg.norm(cart)
    polar[1] = math.atan2(cart[1], cart[0])
    # map output of atan2 to [0,2pi]
    polar[1] = -polar[1] + math.pi if polar[1] < 0 else polar[1]
    return polar


def polar_to_cart(polar, ang):
    polar[1] += ang
    cart = np.array([polar[0] * math.cos(polar[1]), polar[0] * math.sin(polar[1])])
    return cart


def in_fov_check(agent, entity_polar):
    # if agent.fov.dist[0] < entity_polar[0] < agent.fov.dist[1] and \
    # return True
    if entity_polar[0] < agent.fov.dist[1] and \
            math.fabs(entity_polar[1] - agent.state.p_ang) < agent.fov.ang / 2:
        return True
    else:
        return False


def find_grid_id(agent, entity_polar):
    i = None
    j = None
    if in_fov_check(agent, entity_polar):
        i = math.floor((entity_polar[0] - agent.fov.dist[0]) / agent.fov.dist_res)
        j = math.floor((entity_polar[1] - (agent.state.p_ang - agent.fov.ang / 2)) / agent.fov.ang_res)
    return i, j


def collision_check(agent, world, pos=None, size=None):
    if pos is None:
        pos = agent.state.p_pos
    if size is None:
        size = agent.size

    if not (-world.size_x <= pos[0] <= world.size_x and
            -world.size_y <= pos[1] <= world.size_y):
        return True
    for entity in world.entities:
        if entity is not agent and entity.collide and distance_entities(pos, entity) <= entity.size + size:
            return True
    return False


def distance_entities(entity1, entity2):
    pos1 = None
    pos2 = None
    if isinstance(entity1, Entity):
        pos1 = entity1.state.p_pos
    elif isinstance(entity1, np.ndarray):
        pos1 = entity1
    else:
        raise NotImplementedError
    if isinstance(entity2, Entity):
        pos2 = entity2.state.p_pos
    elif isinstance(entity2, np.ndarray):
        pos2 = entity2
    else:
        raise NotImplementedError
    return np.linalg.norm(pos1 - pos2)


def parse_args():
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
    parser.add_argument("--save-rate", type=int, default=20,
                        help="save model once every time this many episodes are completed")
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
    parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2')
    parser.add_argument('--num_timesteps', type=float, default=1e6),
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
    parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--num_env',
                        help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco',
                        default=None, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
    parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--play', default=False, action='store_true')

    parser.add_argument("--debug-display", action="store_true", default=False)
    parser.add_argument('--config_path', help='yaml to load env settings.', default=None, type=str)
    parser.add_argument('--nb_epoch_cycles', type=int, default=3),
    parser.add_argument('--nb_rollout_steps', type=int, default=400),
    parser.add_argument('--nb_epochs', type=int, default=None),
    return parser.parse_args()
