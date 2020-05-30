import baselines.common.tf_util as U
import numpy as np
import tensorflow as tf
from baselines.ddpg.models import Actor, Critic

from multirobot.maddpg.agent import Agent
from multirobot.maddpg.memory import Memory

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class MADDPG(object):
    def __init__(self, env, network,
                 param_noise_n=None,
                 action_noise_n=None,
                 gamma=0.99, tau=0.001, normalize_returns=False, enable_popart=False, normalize_observations=True,
                 batch_size=128, observation_range=(-5., 5.), action_range=(-1., 1.),
                 return_range=(-np.inf, np.inf),
                 critic_l2_reg=0., actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1.,
                 shared_critic=False, **network_kwargs):

        # todo clean the init process later

        nb_actions_n = [action_space.shape[-1] for action_space in env.action_space]

        memory = Memory(limit=int(1e5), action_shape=env.action_space_n_shape,
                        observation_shape=env.observation_space_n_shape, reward_shape=env.reward_shape,
                        terminal_shape=env.terminal_shape)
        critic_n = [Critic(name='critic_%d' % i, network=network, **network_kwargs) for i in
                    range(env.n)] if not shared_critic else [
            Critic(network=network, **network_kwargs)]
        actor_n = [Actor(nb_actions_n[i], name='actor_%d' % i, network=network, **network_kwargs) for i in range(env.n)]

        self.n = env.n
        self.observation_shape = env.observation_space
        self.action_shape = env.action_space
        self.observation_shape_n = env.observation_space_n_shape
        self.action_shape_n = env.action_space_n_shape
        self.reward_scale = reward_scale
        self.memory = memory
        self.batch_size = batch_size
        self.agents = []
        self.normalize_returns = normalize_returns
        self.enable_popart = enable_popart
        self.normalize_observations = normalize_observations
        if shared_critic:
            for i, (actor, param_noise, action_noise, obs_shape, act_shape) in enumerate(
                    zip(actor_n, param_noise_n, action_noise_n, self.observation_shape, self.action_shape)):
                self.agents.append(
                    Agent(actor, critic_n[0], memory, obs_shape.shape, act_shape.shape, self.observation_shape_n,
                          self.action_shape_n, param_noise,
                          action_noise,
                          gamma, tau, normalize_returns, enable_popart, normalize_observations,
                          batch_size, observation_range, action_range, return_range, critic_l2_reg,
                          actor_lr, critic_lr, clip_norm, reward_scale, id=i))
        else:
            for i, (actor, critic, param_noise, action_noise, obs_shape, act_shape) in enumerate(
                    zip(actor_n, critic_n, param_noise_n, action_noise_n, self.observation_shape, self.action_shape)):
                self.agents.append(
                    Agent(actor, critic, memory, obs_shape.shape, act_shape.shape, self.observation_shape_n,
                          self.action_shape_n,
                          param_noise,
                          action_noise,
                          gamma, tau, normalize_returns, enable_popart, normalize_observations,
                          batch_size, observation_range, action_range, return_range, critic_l2_reg,
                          actor_lr, critic_lr, clip_norm, reward_scale, id=i))

        # self.sess_n = [U.single_threaded_session() for _ in self.agents]
        self.observation_range = observation_range

        self.obs0_n = tf.placeholder(tf.float32, shape=(None,) + self.observation_shape_n, name='obs0_n')
        self.obs1_n = tf.placeholder(tf.float32, shape=(None,) + self.observation_shape_n, name='obs1_n')
        self.actions_n = tf.placeholder(tf.float32, shape=(None,) + self.action_shape_n, name='actions_n')

    @property
    def memory_nb_entries(self):
        return self.memory.nb_entries

    def train(self):
        batch = self.memory.sample(batch_size=self.batch_size)

        target_action_n = np.zeros((self.batch_size, self.n, self.action_shape[0].shape[-1]))
        for i, agent in enumerate(self.agents):
            target_action_n[:, i] = agent.compute_target_actions(batch['obs1'][:, i])

        if self.normalize_returns and self.enable_popart:
            raise NotImplementedError
        else:
            target_Q_n = []
            for i, agent in enumerate(self.agents):
                target_Q_n.append(agent.compute_target_Q(batch['obs1'],
                                                         target_action_n,
                                                         batch['rewards'][:, i],
                                                         batch['terminals1'][:, i]))
        cl = np.zeros(self.n)
        al = np.zeros(self.n)
        for i, agent in enumerate(self.agents):
            cl[i], al[i] = agent.get_grads_and_update(batch['obs0'],
                                                      batch['actions'],
                                                      target_Q_n[i])
        return cl, al

    def step(self, obs_n, apply_noise=True, compute_Q=True):
        action_n = np.zeros((self.n, self.action_shape[0].shape[-1]))
        q_n = []
        for i, (agent, obs) in enumerate(zip(self.agents, obs_n)):
            action_n[i], _, _, _ = agent.step(obs, apply_noise=apply_noise, compute_Q=False)
        for i, agent in enumerate(self.agents):
            if compute_Q is True:
                q_n.append(agent.compute_Q(obs_n, action_n))
            else:
                q_n.append(0.)
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
