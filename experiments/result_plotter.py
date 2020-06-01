import os
import re
import time

import numpy as np

import multirobot.scenarios as scenarios
from multirobot.common.saver import Saver
from multirobot.common.util import parse_args
from multirobot.environment.environment import MultiAgentEnv

if __name__ == '__main__':
    arglist = parse_args()

    saver = Saver(arglist)

    results_path, log_path, model_file, actions_path = saver.get_paths()

    # load scenario from script
    scenario = scenarios.load(arglist.scenario).Scenario()
    # create world
    world = scenario.make_world()

    # load the saved landmarks
    if arglist.config_path is not None:
        scenario.load(arglist.config_path, world)

    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None,
                        shared_viewer=True)

    env.render()

    action_cycle_files = os.listdir(actions_path)

    epoch_cycle = []
    for action_cycle in action_cycle_files:
        s = re.split('[-.]', action_cycle)
        epoch_cycle.append([int(s[0]), int(s[1])])
    latest = sorted(epoch_cycle, key=lambda x: x[0])
    latest = sorted(latest, key=lambda x: x[1])
    latest_file = os.path.join(actions_path, '%04d-%04d.csv' % (latest[-1][0], latest[-1][1]))
    latest_actions = np.loadtxt(latest_file, delimiter=',')

    assert latest_actions.shape[-1] % 2 == 0
    n=latest_actions.shape[-1]//2
    for action in latest_actions:
        action=action.reshape(n,2)
        env.step(action)
        env.render()
        time.sleep(0.1)
