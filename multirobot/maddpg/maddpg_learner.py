from copy import copy
from functools import reduce

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from baselines import logger
from baselines.common.mpi_adam import MpiAdam
import baselines.common.tf_util as U
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.ddpg.ddpg import DDPG

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / (stats.std + 1e-8)


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean


def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keepdims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


def get_target_updates(vars, target_vars, tau):
    logger.info('setting up target updates ...')
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
        logger.info('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)


def get_perturbed_actor_updates(actor, perturbed_actor, param_noise_stddev):
    assert len(actor.vars) == len(perturbed_actor.vars)
    assert len(actor.perturbable_vars) == len(perturbed_actor.perturbable_vars)

    updates = []
    for var, perturbed_var in zip(actor.vars, perturbed_actor.vars):
        if var in actor.perturbable_vars:
            logger.info('  {} <- {} + noise'.format(perturbed_var.name, var.name))
            updates.append(
                tf.assign(perturbed_var, var + tf.random_normal(tf.shape(var), mean=0., stddev=param_noise_stddev)))
        else:
            logger.info('  {} <- {}'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var))
    assert len(updates) == len(actor.vars)
    return tf.group(*updates)


class Agent(DDPG):
    def __init__(self, actor, critic, memory, observation_shape, action_shape, param_noise=None, action_noise=None,
                 gamma=0.99, tau=0.001, normalize_returns=False, enable_popart=False, normalize_observations=True,
                 batch_size=128, observation_range=(-5., 5.), action_range=(-1., 1.),
                 return_range=(-np.inf, np.inf),
                 critic_l2_reg=0., actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1.,
                 id=None):

        assert id is not None
        # Inputs.
        self.obs0 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs0')
        self.obs1 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs1')
        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
        self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
        self.actions = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='actions')
        self.critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')
        self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')

        # Parameters.
        self.gamma = gamma
        self.tau = tau
        self.memory = memory
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.action_noise = action_noise
        self.param_noise = param_noise
        self.action_range = action_range
        self.return_range = return_range
        self.observation_range = observation_range
        self.critic = critic
        self.actor = actor
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_norm = clip_norm
        self.enable_popart = enable_popart
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.stats_sample = None
        self.critic_l2_reg = critic_l2_reg

        # Observation normalization.
        if self.normalize_observations:
            with tf.variable_scope('agent_%d/obs_rms' % id):
                self.obs_rms = RunningMeanStd(shape=observation_shape)
        else:
            self.obs_rms = None
        normalized_obs0 = tf.clip_by_value(normalize(self.obs0, self.obs_rms),
                                           self.observation_range[0], self.observation_range[1])
        normalized_obs1 = tf.clip_by_value(normalize(self.obs1, self.obs_rms),
                                           self.observation_range[0], self.observation_range[1])

        # Return normalization.
        if self.normalize_returns:
            with tf.variable_scope('agent_%d/ret_rms' % id):
                self.ret_rms = RunningMeanStd()
        else:
            self.ret_rms = None

        # Create target networks.
        target_actor = copy(actor)
        target_actor.name = 'target_actor'
        self.target_actor = target_actor
        target_critic = copy(critic)
        target_critic.name = 'target_critic'
        self.target_critic = target_critic

        # Create networks and core TF parts that are shared across setup parts.
        self.actor_tf = actor(normalized_obs0)
        self.normalized_critic_tf = critic(normalized_obs0, self.actions)
        self.critic_tf = denormalize(
            tf.clip_by_value(self.normalized_critic_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
        self.normalized_critic_with_actor_tf = critic(normalized_obs0, self.actor_tf, reuse=True)
        self.critic_with_actor_tf = denormalize(
            tf.clip_by_value(self.normalized_critic_with_actor_tf, self.return_range[0], self.return_range[1]),
            self.ret_rms)
        Q_obs1 = denormalize(target_critic(normalized_obs1, target_actor(normalized_obs1)), self.ret_rms)
        self.target_Q = self.rewards + (1. - self.terminals1) * gamma * Q_obs1

        # Set up parts.
        if self.param_noise is not None:
            self.setup_param_noise(normalized_obs0)
        self.setup_actor_optimizer()
        self.setup_critic_optimizer()
        if self.normalize_returns and self.enable_popart:
            self.setup_popart()
        self.setup_stats()
        self.setup_target_network_updates()

        self.initial_state = None  # recurrent architectures not supported yet

        self.id = id

    def obs_rms_update(self, obs0):
        if self.normalize_observations:
            self.obs_rms.update(np.array([obs0]))


class MADDPG(object):

    def __init__(self, actor_n, critic_n, memory, observation_shape, action_shape, param_noise_n=None,
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
                self.agents.append(Agent(actor, critic_n[0], memory, obs_shape.shape, act_shape.shape, param_noise,
                                         action_noise,
                                         gamma, tau, normalize_returns, enable_popart, normalize_observations,
                                         batch_size, observation_range, action_range, return_range, critic_l2_reg,
                                         actor_lr, critic_lr, clip_norm, reward_scale, id=i))
        else:
            for i, (actor, critic, param_noise, action_noise, obs_shape, act_shape) in enumerate(
                    zip(actor_n, critic_n, param_noise_n, action_noise_n, observation_shape, action_shape)):
                self.agents.append(Agent(actor, critic, memory, obs_shape.shape, act_shape.shape, param_noise,
                                         action_noise,
                                         gamma, tau, normalize_returns, enable_popart, normalize_observations,
                                         batch_size, observation_range, action_range, return_range, critic_l2_reg,
                                         actor_lr, critic_lr, clip_norm, reward_scale, id=i))

        # self.sess_n = [U.single_threaded_session() for _ in self.agents]

    def train(self):
        for agent in self.agents:
            agent.train()

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
