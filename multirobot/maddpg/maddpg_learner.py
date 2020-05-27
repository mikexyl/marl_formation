import baselines.common.tf_util as U
import numpy as np

from multirobot.maddpg.agent import Agent

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class MADDPG(object):

    def __init__(self, actor_n, critic_n, memory, observation_shape, action_shape, observation_shape_n, action_shape_n,
                 param_noise_n=None,
                 action_noise_n=None,
                 gamma=0.99, tau=0.001, normalize_returns=False, enable_popart=False, normalize_observations=True,
                 batch_size=128, observation_range=(-5., 5.), action_range=(-1., 1.),
                 return_range=(-np.inf, np.inf),
                 critic_l2_reg=0., actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1.,
                 shared_critic=False):
        assert (shared_critic and len(critic_n) == 1) or (not shared_critic and len(critic_n) == len(actor_n))

        self.n = len(actor_n)
        self.reward_scale = reward_scale
        self.memory = memory
        self.agents = []
        if shared_critic:
            for i, (actor, param_noise, action_noise, obs_shape, act_shape) in enumerate(
                    zip(actor_n, param_noise_n, action_noise_n, observation_shape, action_shape)):
                self.agents.append(
                    Agent(actor, critic_n[0], memory, obs_shape.shape, act_shape.shape, observation_shape_n,
                          action_shape_n, param_noise,
                          action_noise,
                          gamma, tau, normalize_returns, enable_popart, normalize_observations,
                          batch_size, observation_range, action_range, return_range, critic_l2_reg,
                          actor_lr, critic_lr, clip_norm, reward_scale, id=i))
        else:
            for i, (actor, critic, param_noise, action_noise, obs_shape, act_shape) in enumerate(
                    zip(actor_n, critic_n, param_noise_n, action_noise_n, observation_shape, action_shape)):
                self.agents.append(
                    Agent(actor, critic, memory, obs_shape.shape, act_shape.shape, observation_shape_n, action_shape_n,
                          param_noise,
                          action_noise,
                          gamma, tau, normalize_returns, enable_popart, normalize_observations,
                          batch_size, observation_range, action_range, return_range, critic_l2_reg,
                          actor_lr, critic_lr, clip_norm, reward_scale, id=i))

        # self.sess_n = [U.single_threaded_session() for _ in self.agents]

    def train(self):
        cl = np.zeros(self.n)
        al = np.zeros(self.n)
        for i, agent in enumerate(self.agents):
            cl[i], al[i] = agent.train()
        return cl, al

    def step(self, obs_n, apply_noise=True, compute_Q=True):
        action_n = []
        q_n = []
        for i, (agent, obs) in enumerate(zip(self.agents, obs_n)):
            action, q, _, _ = agent.step(obs, apply_noise=apply_noise, compute_Q=compute_Q)
            action_n.append(action)
            q_n.append(q)
        return action_n, q_n, None, None

    def store_transition(self, obs0_n, action_n, reward_n, obs1_n, terminal1_n):
        reward_n *= self.reward_scale
        # todo need to reshape or not
        self.memory.append(obs0_n, action_n, reward_n, obs1_n, terminal1_n)
        for i, agent in enumerate(self.agents):
            agent.obs_rms_update(obs0_n[i])

    def initialize(self):
        # todo only single thread single sess for now
        for agent in self.agents:
            sess = U.get_session()
            agent.initialize(sess)

    def update_target_net(self):
        for agent in self.agents:
            agent.update_target_net()

    def get_stats(self):
        return [agent.get_stats() for agent in self.agents]

    def adapt_param_noise(self):
        for agent in self.agents:
            agent.adapt_param_noise()

    def reset(self, agent_id=None):
        if agent_id is not None:
            self.agents[agent_id].reset()
        else:
            for agent in self.agents:
                agent.reset()
