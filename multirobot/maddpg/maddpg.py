import os
import time
from collections import deque
import pickle

import baselines.common.tf_util as U
import numpy as np
from baselines import logger
from baselines.common import set_global_seeds
from baselines.ddpg.ddpg_learner import DDPG
from baselines.ddpg.memory import Memory
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise

from glog import info

# disable mpi for now
# try:
#     from mpi4py import MPI
# except ImportError:
#     MPI = None
MPI = None


def learn(network, env,
          seed=None,
          total_timesteps=None,
          nb_epochs=None,  # with default settings, perform 1M steps total
          nb_epoch_cycles=20,
          nb_rollout_steps=100,
          reward_scale=1.0,
          render=False,
          render_eval=False,
          noise_type='adaptive-param_0.2',
          normalize_returns=False,
          normalize_observations=True,
          critic_l2_reg=1e-2,
          actor_lr=1e-4,
          critic_lr=1e-3,
          popart=False,
          gamma=0.99,
          clip_norm=None,
          nb_train_steps=50,  # per epoch cycle and MPI worker,
          nb_eval_steps=100,
          batch_size=64,  # per MPI worker
          tau=0.01,
          eval_env=None,
          param_noise_adaption_interval=50,
          shared_critic=True,
          **network_kwargs):
    set_global_seeds(seed)

    # multi robot env check
    assert len(env.action_space) == env.n and len(env.observation_space) == env.n

    if total_timesteps is not None:
        assert nb_epochs is None
        nb_epochs = int(total_timesteps) // (nb_epoch_cycles * nb_rollout_steps)
    else:
        nb_epochs = 500

    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0

    nb_actions = [action_space.shape[-1] for action_space in env.action_space]
    assert np.array([np.abs(action_space.low) == action_space.high for action_space in env.action_space]).all()
    # assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.

    # todo
    memory = Memory(limit=int(1e6), action_shape=env.action_space[0].shape,
                    observation_shape=env.observation_space[0].shape)

    critics = None
    if shared_critic is True:
        critics = [Critic(network=network, **network_kwargs)]
    else:
        critics = [Critic(network=network, **network_kwargs) for _ in range(env.n)]

    actors = [Actor(nb_actions[i], network=network, **network_kwargs) for i in range(env.n)]

    action_noise = None
    param_noise = None
    if noise_type is not None:
        for current_noise_type in noise_type.split(','):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                                            sigma=float(stddev) * np.ones(nb_actions))
            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    max_actions = np.array([action_space.high for action_space in env.action_space])
    logger.info('scaling actions by {} before executing in env'.format(max_actions))

    if shared_critic:
        agents = [DDPG(actors[i], critics[0], memory, env.observation_space[i].shape, env.action_space[i].shape,
                       gamma=gamma, tau=tau, normalize_returns=normalize_returns,
                       normalize_observations=normalize_observations,
                       batch_size=batch_size, action_noise=action_noise, param_noise=param_noise,
                       critic_l2_reg=critic_l2_reg,
                       actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
                       reward_scale=reward_scale) for i in range(env.n)]
    else:
        agents = [DDPG(actors[i], critics[i], memory, env.observation_space[i], env.action_space[i],
                       gamma=gamma, tau=tau, normalize_returns=normalize_returns,
                       normalize_observations=normalize_observations,
                       batch_size=batch_size, action_noise=action_noise, param_noise=param_noise,
                       critic_l2_reg=critic_l2_reg,
                       actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
                       reward_scale=reward_scale) for i in range(env.n)]
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()) for agent in agents)

    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)

    sess = U.get_session()
    # todo this looks not right
    for agent in agents:
        agent.initialize(sess)
        sess.graph.finalize()

        agent.reset()

    obs_n = env.reset()
    if eval_env is not None:
        eval_obs = eval_env.reset()
    nb_agents = env.n
    episode_reward = np.zeros(nb_agents, dtype=np.float32)  # vector
    episode_step = np.zeros(nb_agents, dtype=int)  # vector
    episodes = 0
    t = 0
    epoch = 0

    start_time = time.time()

    epoch_episode_rewards = []
    epoch_episode_steps = []
    epoch_actions = []
    epoch_qs = []
    epoch_episodes = 0
    for epoch in range(nb_epochs):
        for cycle in range(nb_epoch_cycles):
            info("epoch %d, cycle %d", epoch, cycle)
            # Perform rollouts.
            for t_rollout in range(nb_rollout_steps):
                # Predict next action.
                # action_n, q_n, _, _ = np.split(
                #     np.array([agent.step(obs_n, apply_noise=True, compute_Q=True) for agent in agents]), 2)
                action_n = []
                q_n = []
                for agent in agents:
                    action, q, _, _ = agent.step(obs_n, apply_noise=True, compute_Q=True)
                    action_n.append(action)
                    q_n.append(q)

                # Execute next action.
                if rank == 0 and render:
                    env.render()

                new_obs_n, r_n, done_n, info_n = env.step(
                    max_actions * action_n)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])

                t += 1
                episode_reward += r_n
                episode_step += 1

                # Book-keeping.
                epoch_actions.append(action_n)
                epoch_qs.append(q_n)
                # todo centralized stroe?
                for agent in agents:
                    agent.store_transition(obs_n, action_n, r_n, new_obs_n,
                                           done_n)  # the batched data will be unrolled in memory.py's append.

                obs_n = new_obs_n

                for d in range(len(done_n)):
                    if done_n[d]:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward[d])
                        episode_rewards_history.append(episode_reward[d])
                        epoch_episode_steps.append(episode_step[d])
                        # todo these resets doubtful
                        episode_reward[d] = 0.
                        episode_step[d] = 0
                        epoch_episodes += 1
                        episodes += 1
                        # todo if this resets the centralized critic?
                        for agent in agents:
                            agent.reset()

            # train
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_adaptive_distances = []
            for t_train in range(nb_train_steps):
                # Adapt param noise, if necessary.
                if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                    for agent in agents:
                        distance = agent.adapt_param_noise()
                        epoch_adaptive_distances.append(distance)

            cl_n = []
            al_n = []
            for agent in agents:
                cl, al = agent.train()
                cl_n.append(cl)
                al_n.append(al)

                epoch_critic_losses.append(cl)
                epoch_actor_losses.append(al)
                agent.update_target_net()

                # Evaluate
                # later

        if MPI is not None:
            mpi_size = MPI.COMM_WORLD.Get_size()
        else:
            mpi_size = 1

        # todo logger later
        # Log stats.
        # XXX shouldn't call np.mean on variable length lists
        duration = time.time() - start_time
        combined_stats = {}
        combined_stats['multi/n'] = env.n
        combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
        combined_stats['rollout/return_std'] = np.std(epoch_episode_rewards)
        combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
        combined_stats['rollout/return_history_std'] = np.std(episode_rewards_history)
        combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
        combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
        combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
        combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
        combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
        combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
        combined_stats['total/duration'] = duration
        combined_stats['total/steps_per_second'] = float(t) / float(duration)
        combined_stats['total/episodes'] = episodes
        combined_stats['rollout/episodes'] = epoch_episodes
        combined_stats['rollout/actions_std'] = np.std(epoch_actions)
        # Evaluation statistics.
        if eval_env is not None:
            # later
            pass

        combined_stats_sums = np.array([np.array(x).flatten()[0] for x in combined_stats.values()])
        if MPI is not None:
            combined_stats_sums = MPI.COMM_WORLD.allreduce(combined_stats_sums)

        combined_stats = {k: v / mpi_size for (k, v) in zip(combined_stats.keys(), combined_stats_sums)}

        # Total statistics.
        combined_stats['total/epochs'] = epoch + 1
        combined_stats['total/steps'] = t

        for key in sorted(combined_stats.keys()):
            logger.record_tabular(key, combined_stats[key])

        if rank == 0:
            logger.dump_tabular()
        logger.info('')
        logdir = logger.get_dir()
        if rank == 0 and logdir:
            if hasattr(env, 'get_state'):
                with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                    pickle.dump(env.get_state(), f)
            if eval_env and hasattr(eval_env, 'get_state'):
                with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                    pickle.dump(eval_env.get_state(), f)
