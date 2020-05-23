from gym import Env as GymEnv

from multirobot.environment import MultiAgentEnv as MaddpgEnv


class DdpgEnv(GymEnv):
    def __init__(self, maddpg_env):
        if not isinstance(maddpg_env, MaddpgEnv):
            raise NotImplementedError
        elif maddpg_env.n != 1:
            raise NotImplementedError
        else:
            super(DdpgEnv, self).__init__()
            self.maddpg_env = maddpg_env
            self.action_space=maddpg_env.action_space[0]
            self.observation_space=maddpg_env.observation_space[0]

    def step(self, action):
        action_n = [action]
        obs_n, rew_n, done_n, info_n = self.maddpg_env.step(action_n)
        return obs_n[0], rew_n[0], done_n[0], info_n[0]

    def reset(self):
        obs_n = self.maddpg_env.reset()
        return obs_n[0]

    def render(self, mode='human'):
        self.maddpg_env.render()

    def close(self):
        pass
