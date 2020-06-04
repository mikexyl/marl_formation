from baselines.common.vec_env import VecEnv

from multirobot.maddpg.environment import BlMaddpgEnv


# todo only support single env
class MarlVecEnv(VecEnv):
    def __init__(self, env):
        assert isinstance(env, BlMaddpgEnv)
        self.env = env
        super(MarlVecEnv, self).__init__(1, env.observation_space, env.action_space)
        # obs_space = env.observation_space
        # self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.actions_n = None

    def step_async(self, actions_n):
        self.actions_n = actions_n

    def step_wait(self):
        return self.env.step(self.actions_n)

    def reset(self):
        return self.env.reset()

    def get_images(self):
        return self.env.render(mode='rgb_array')

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)
