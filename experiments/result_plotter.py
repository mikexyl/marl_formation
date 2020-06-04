import os
import re
import time
from collections import deque

import numpy as np

import multirobot.scenarios as scenarios
from multirobot.common.saver import Saver
from multirobot.common.cmd_util import parse_args, make_base_env
from multirobot.environment.environment import MultiAgentEnv
from multirobot.maddpg.environment import BlMaddpgEnv

if __name__ == '__main__':
    arglist = parse_args()

    saver = Saver(arglist)

    results_path, log_path, model_file, actions_path = saver.get_paths()

    env = make_base_env(arglist.scenario, arglist, arglist.benchmark)
    ddpg_env = BlMaddpgEnv(env)

    env.render()

    action_cycle_files = os.listdir(actions_path)

    epoch_cycle = []
    for action_cycle in action_cycle_files:
        s = re.split('[-.]', action_cycle)
        epoch_cycle.append([int(s[0]), int(s[1])])
    latest = sorted(epoch_cycle, key=lambda x: x[0])
    latest = sorted(latest, key=lambda x: x[1])
    latest_file = os.path.join(actions_path, '%04d-%04d.csv' % (latest[-1][0], latest[-1][1]))
    print(latest_file)
    latest_actions = np.loadtxt(latest_file, delimiter=',')

    assert latest_actions.shape[-1] % 2 == 0
    n = latest_actions.shape[-1] // 2
    episode_reward = np.zeros((n, 1), dtype=np.float32)  # vector
    epoch_episode_rewards = [[] for _ in range(env.n)]
    episode_rewards_history = [deque(maxlen=100) for _ in range(env.n)]
    epoch_episodes = 0
    episodes = 0  # scalar

    for action in latest_actions:
        action = action.reshape(n, 2)
        _, r, done, _ = env.step(action)
        episode_reward += r
        if any(done):
            env.reset()
            for d in range(len(done)):
                if done[d]:
                    # Episode done.
                    epoch_episode_rewards[d].append(episode_reward[d][0])
                    episode_rewards_history[d].append(episode_reward[d][0])
                    episode_reward[d] = 0.
                    epoch_episodes += 1
                    episodes += 1
        env.render()
        # time.sleep(0.1)
    pass
