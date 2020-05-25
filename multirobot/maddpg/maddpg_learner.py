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
        super(Agent, self).__init__(actor, critic, memory, observation_shape, action_shape, param_noise,
                                    action_noise,
                                    gamma, tau, normalize_returns, enable_popart, normalize_observations,
                                    batch_size, observation_range, action_range, return_range, critic_l2_reg,
                                    actor_lr, critic_lr, clip_norm, reward_scale)
        self.id = id


class MADDPG(object):

    def __init__(self, actor_n, critic_n, memory, observation_shape, action_shape, param_noise=None, action_noise=None,
                 gamma=0.99, tau=0.001, normalize_returns=False, enable_popart=False, normalize_observations=True,
                 batch_size=128, observation_range=(-5., 5.), action_range=(-1., 1.),
                 return_range=(-np.inf, np.inf),
                 critic_l2_reg=0., actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1.,
                 shared_critic=False):

        assert (shared_critic and len(critic_n) == 1) or (not shared_critic and len(critic_n) == len(actor_n))

        self.n = len(actor_n)

        self.agents = [Agent(actor, critic, memory, observation_shape, action_shape, param_noise,
                             action_noise,
                             gamma, tau, normalize_returns, enable_popart, normalize_observations,
                             batch_size, observation_range, action_range, return_range, critic_l2_reg,
                             actor_lr, critic_lr, clip_norm, reward_scale, id=i) for i, actor, critic in
                       enumerate(zip(actor_n, critic_n))]
        
        # Inputs.
        self.obs0 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs0')
        self.obs1 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs1')
        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
        self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
        self.actions = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='actions')
        self.critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')
        self.param_noise_stddev_n = [tf.placeholder(tf.float32, shape=(), name='param_noise_stddev') for _ in actor_n]

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
        self.critic_n = critic_n
        self.actor_n = actor_n
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_norm = clip_norm
        self.enable_popart = enable_popart
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.stats_sample = None
        self.critic_l2_reg = critic_l2_reg

        # Observation normalization.
        # todo not implemented yet
        if self.normalize_observations:
            # raise NotImplementedError
            self.obs_rms_n = []
            for i in range(self.n):
                with tf.variable_scope('obs_rms_%d' % 1):
                    self.obs_rms_n.append(RunningMeanStd(shape=observation_shape))
        else:
            self.obs_rms_n = None

        normalized_obs0 = tf.clip_by_value(normalize(self.obs0, self.obs_rms_n),
                                           self.observation_range[0], self.observation_range[1])
        normalized_obs1 = tf.clip_by_value(normalize(self.obs1, self.obs_rms_n),
                                           self.observation_range[0], self.observation_range[1])

        # Return normalization.
        if self.normalize_returns:
            self.ret_rms_n = []
            for i in range(self.n):
                with tf.variable_scope('ret_rms_%d' % 1):
                    self.ret_rms_n.append(RunningMeanStd())
        else:
            self.ret_rms_n = None

        # Create target networks.
        # now its multi agent version
        target_actor_n = []
        for i, actor in enumerate(actor_n):
            target_actor = copy(actor)
            target_actor.name = 'target_actor_%d' % i
            target_actor_n.append(target_actor)
        self.target_actor_n = target_actor_n

        target_critics_n = []
        for i, critic in enumerate(critic_n):
            target_critic = copy(critic)
            target_critic.name = 'target_critic_%d' % i
            target_critics_n.append(target_critic)
        self.target_critics_n = target_critics_n

        # Create networks and core TF parts that are shared across setup parts.
        self.actor_tf_n = [actor_n(normalized_obs0) for _ in range(self.n)]
        self.normalized_critic_tf_n = [critic_n(normalized_obs0, self.actions) for _ in range(self.n)]
        self.critic_tf_n = [denormalize(
            tf.clip_by_value(normalized_critic_tf, self.return_range[0], self.return_range[1]), ret_rms)
            for normalized_critic_tf, ret_rms in zip(self.normalized_critic_tf_n, self.ret_rms_n)]
        self.normalized_critic_with_actor_tfs = [critic_n(normalized_obs0, actor_tf, reuse=True) for actor_tf in
                                                 self.actor_tf_n]
        self.critic_with_actor_tf_n = [denormalize(
            tf.clip_by_value(normalized_critic_with_actor_tf, self.return_range[0], self.return_range[1]),
            ret_rms) for normalized_critic_with_actor_tf, ret_rms in
            zip(self.normalized_critic_with_actor_tfs, self.ret_rms_n)]
        Q_obs1_n = [denormalize(target_critic(normalized_obs1, target_actor(normalized_obs1)), ret_rms) for
                    target_critic, target_actor, ret_rms in
                    zip(self.target_critics_n, self.target_actor_n, self.ret_rms_n)]
        self.target_Q_n = self.rewards + (1. - self.terminals1) * gamma * Q_obs1_n

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

    def setup_param_noise(self, normalized_obs0):
        assert self.param_noise is not None

        logger.info('setting up param noise')

        # Configure perturbed actor.
        param_noise_actor_n = copy(self.actor_n)
        self.perturbed_actor_tf_n = []
        self.perturb_policy_ops_n = []
        for i, param_noise_actor in enumerate(param_noise_actor_n):
            param_noise_actor.name = 'param_noise_actor_%d' % i
            self.perturbed_actor_tf_n.append(param_noise_actor(normalized_obs0))
            self.perturb_policy_ops_n.append(get_perturbed_actor_updates(self.actor_n[i], param_noise_actor,
                                                                         self.param_noise_stddev_n[i]))

        # Configure separate copy for stddev adoption.
        adaptive_param_noise_actor_n = copy(self.actor_n)
        self.perturb_adaptive_policy_ops_n = []
        for i, adaptive_param_noise_actor in enumerate(adaptive_param_noise_actor_n):
            adaptive_param_noise_actor.name = 'adaptive_param_noise_actor_%d' % i
            adaptive_actor_tf = adaptive_param_noise_actor(normalized_obs0)
            self.perturb_adaptive_policy_ops_n.append(
                get_perturbed_actor_updates(self.actor_n[i], adaptive_param_noise_actor,
                                            self.param_noise_stddev_n[i]))
            self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.actor_tf_n[i] - adaptive_actor_tf)))

    def setup_actor_optimizer(self):
        logger.info('setting up actor optimizer')
        self.actor_loss_n = [-tf.reduce_mean(critic_with_actor_tf) for critic_with_actor_tf in
                             self.critic_with_actor_tf_n]
        actor_shapes_n = [[var.get_shape().as_list() for var in actor.trainable_vars] for actor in self.actor_n]
        actor_nb_params_n = [sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes]) for actor_shapes in
                             actor_shapes_n]
        for actor_shapes, actor_nb_params in zip(actor_shapes_n, actor_nb_params_n):
            logger.info('  actor shapes: {}'.format(actor_shapes))
            logger.info('  actor params: {}'.format(actor_nb_params))
        self.actor_grads_n = [U.flatgrad(actor_loss, actor.trainable_vars, clip_norm=self.clip_norm) for
                              actor_loss, actor in zip(self.actor_loss_n, self.actor_n)]
        self.actor_optimizer_n = [MpiAdam(var_list=actor.trainable_vars,
                                          beta1=0.9, beta2=0.999, epsilon=1e-08) for actor in self.actor_n]
