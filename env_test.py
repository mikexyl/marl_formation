import argparse

import numpy as np

import multirobot.scenarios as scenarios
from multirobot.environment import MultiAgentEnv, make_env

import yaml
import os

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()

    # load the saved landmarks
    # scenario.save(world) -->used to save fixed landmarks
    flie_path = '/home/zhonvsky/venv/marl_formation/multirobot/scenarios/scenario_P_pos.yaml'
    scenario.load(flie_path,world)

    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None,
                        shared_viewer=True)

    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    # policies = [InteractivePolicy(env, i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    while True:
        # query for action from each agent's policy
        act_n = []
        for i in range(env.n):
            # act_n.append(policy.action(obs_n[i]))
            # let it do nothing
            act_n.append(np.array([-0.4, -1]))
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        # render all agent views
        env.render()
        # display rewards
        # for agent in env.world.agents:
        #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))

