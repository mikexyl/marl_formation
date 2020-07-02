from baselines.common.vec_env import VecEnv

from multirobot.maddpg.environment import BlMaddpgEnv


# todo only support single env
class MarlVecEnv(VecEnv):
    """
    wrap multirobot.maddpg.environment.BlMaddpgEnv as a child class of baselines.common.vec_env.VecEnv,
    to be compatible with vec_video_recorder to record video step_async and step_wait are the biggest difference. In
    this case, step_async stores actions temporarily, and step_wait executes the actions. This is for get images and
    record video, causing no difference to training process
    """

    def __init__(self, env):
        """

        @param env: multirobot.maddpg.environment.BlMaddpgEnv to be wrapped
        """
        assert isinstance(env, BlMaddpgEnv)
        self.env = env
        super(MarlVecEnv, self).__init__(1, env.observation_space, env.action_space)
        # obs_space = env.observation_space
        # self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.actions_n = None

    def step_async(self, actions_n):
        """
        store actions of all vehicles
        @param actions_n: actions for vehicles
        @return:
        """
        self.actions_n = actions_n

    def step_wait(self):
        """
        execute stored actions
        @return: step return of env
        """
        return self.env.step(self.actions_n)

    def reset(self):
        """
        reset self.env
        @return: reset return of env
        """
        return self.env.reset()

    def get_images(self):
        """
        get image, to add to video later
        @return: rendered image
        """
        return self.env.render(mode='rgb_array')

    def render(self, mode='human'):
        """
        render the scenario on screen
        @param mode: mode of render, defaultly 'human
        @return: result of render
        """
        return self.env.render(mode=mode)

    def __getattr__(self, name):
        """
        get attribute of private attribute or attribute of env
        @param name:
        @return: attribute
        """
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)
