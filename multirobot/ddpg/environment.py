from gym import Env as GymEnv

from multirobot.environment import MultiAgentEnv as MaddpgEnv
import numpy as np


class DdpgEnv(GymEnv):
    def __init__(self, maddpg_env):
        if not isinstance(maddpg_env, MaddpgEnv):
            raise NotImplementedError
        else:
            super(DdpgEnv, self).__init__()
            self.maddpg_env = maddpg_env
            self.action_space = maddpg_env.action_space
            self.observation_space = maddpg_env.observation_space
            self.n = self.maddpg_env.n
            self.action_space_n_shape = maddpg_env.action_space[0].shape[-1] * self.n
            self.observation_space_n_shape = maddpg_env.observation_space[0].shape[-1] * self.n

    def step(self, action_n):
        action_n = [action.reshape(2) for action in action_n]
        obs_n, rew_n, done_n, info_n = self.maddpg_env.step(action_n)
        if done_n[0]:
            pass
        return self.obs_reshape(obs_n), np.array(rew_n), np.array(done_n), np.array(info_n)

    def reset(self):
        obs_n = self.maddpg_env.reset()
        return self.obs_reshape(obs_n)

    def render(self, mode='human'):
        self.maddpg_env.render()

    def close(self):
        pass

    @staticmethod
    def obs_reshape(obs_n):
        return np.vstack([obs.reshape(1, obs.shape[0]) for obs in obs_n])
