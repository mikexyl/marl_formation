import os
import pickle
import time
from collections import deque

import numpy as np
from baselines import logger
from baselines.common import set_global_seeds
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise

from multirobot.maddpg.maddpg_learner import MADDPG
from multirobot.maddpg.memory import Memory

try:
    from mpi4py import MPI
except ImportError:
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

    if total_timesteps is not None:
        assert nb_epochs is None
        nb_epochs = int(total_timesteps) // (nb_epoch_cycles * nb_rollout_steps)
    else:
        nb_epochs = 500

    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0

    nb_actions_n = [action_space.shape[-1] for action_space in env.action_space]
    assert np.array([(np.abs(action_space.low) == action_space.high) for action_space in
                     env.action_space]).all()  # we assume symmetric actions.

    memory = Memory(limit=int(1e6), action_shape=env.action_space_n_shape,
                    observation_shape=env.observation_space_n_shape, reward_shape=env.reward_shape,
                    terminal_shape=env.terminal_shape)
    critic_n = [Critic(network=network, **network_kwargs) for _ in range(env.n)] if not shared_critic else [
        Critic(network=network, **network_kwargs)]
    actor_n = [Actor(nb_actions_n[i], network=network, **network_kwargs) for i in range(env.n)]

    action_noise_n = []
    param_noise_n = []
    if noise_type is not None:
        for current_noise_type in noise_type.split(','):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                param_noise_n = [
                    AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev)) for _ in
                    range(env.n)]
                action_noise_n = [None for _ in range(env.n)]
            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                param_noise_n = [None for _ in range(env.n)]
                action_noise_n = [NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
                                  for nb_actions in nb_actions_n]
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                param_noise_n = [None for _ in range(env.n)]
                action_noise_n = [OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                                               sigma=float(stddev) * np.ones(nb_actions)) for nb_actions
                                  in nb_actions_n]
            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    max_action_n = [action_space.high for action_space in env.action_space]
    logger.info('scaling actions by {} before executing in env'.format(max_action_n))

    agent = MADDPG(actor_n, critic_n, memory, env.observation_space, env.action_space, env.observation_space_n_shape,
                   env.action_space_n_shape,
                   gamma=gamma, tau=tau, normalize_returns=normalize_returns,
                   normalize_observations=normalize_observations,
                   batch_size=batch_size, action_noise_n=action_noise_n, param_noise_n=param_noise_n,
                   critic_l2_reg=critic_l2_reg,
                   actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
                   reward_scale=reward_scale, shared_critic=shared_critic)

    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = [deque(maxlen=100) for _ in range(env.n)]
    # sess = U.get_session()
    # Prepare everything.
    agent.initialize()
    # sess.graph.finalize()

    agent.reset()

    obs = env.reset()
    if eval_env is not None:
        eval_obs = eval_env.reset()
    nveh = obs.shape[0]

    episode_reward = np.zeros((nveh, 1), dtype=np.float32)  # vector
    episode_step = np.zeros(nveh, dtype=int)  # vector
    episodes = 0  # scalar
    t = 0  # scalar

    epoch = 0

    start_time = time.time()

    epoch_episode_rewards = [[] for _ in range(env.n)]
    epoch_episode_steps = [[] for _ in range(env.n)]
    epoch_actions = [[] for _ in range(env.n)]
    epoch_qs = [[] for _ in range(env.n)]
    epoch_episodes = 0
    for epoch in range(nb_epochs):
        for cycle in range(nb_epoch_cycles):
            from glog import info
            info("epoch %d, cycle %d" % (epoch, cycle))
            # Perform rollouts.
            # if nveh > 1:
            #     # if simulating multiple envs in parallel, impossible to reset agent at the end of the episode in each
            #     # of the environments, so resetting here instead
            #     agent.reset()
            for t_rollout in range(nb_rollout_steps):
                # Predict next action.
                action_n, q_n, _, _ = agent.step(obs, apply_noise=True, compute_Q=True)

                # Execute next action.
                if rank == 0 and render:
                    env.render()

                # max_action is of dimension A, whereas action is dimension (nenvs, A) - the multiplication gets broadcasted to the batch
                # todo max_action not scale yet
                new_obs, r, done, info = env.step(
                    action_n)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                # note these outputs are batched from vecenv

                t += 1
                if rank == 0 and render:
                    env.render()
                episode_reward += r
                episode_step += 1

                # Book-keeping.
                for i in range(env.n):
                    epoch_actions[i].append(action_n[i])
                    epoch_qs[i].append(q_n[i])
                agent.store_transition(obs, action_n, r, new_obs,
                                       done)  # the batched data will be unrolled in memory.py's append.

                obs = new_obs

                for d in range(len(done)):
                    if done[d]:
                        # Episode done.
                        epoch_episode_rewards[d].append(episode_reward[d][0])
                        episode_rewards_history[d].append(episode_reward[d][0])
                        epoch_episode_steps[d].append(episode_step[d])
                        episode_reward[d] = 0.
                        episode_step[d] = 0
                        epoch_episodes += 1
                        episodes += 1
                        agent.reset(d)
                        env.reset_vehicle(d)

            # Train.
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_adaptive_distances = []
            for t_train in range(nb_train_steps):
                # Adapt param noise, if necessary.
                if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                    distance = agent.adapt_param_noise()
                    epoch_adaptive_distances.append(distance)

                cl, al = agent.train()
                epoch_critic_losses.append(cl)
                epoch_actor_losses.append(al)
                agent.update_target_net()

            # Evaluate.
            eval_episode_rewards = []
            eval_qs = []
            if eval_env is not None:
                nenvs_eval = eval_obs.shape[0]
                eval_episode_reward = np.zeros(nenvs_eval, dtype=np.float32)
                for t_rollout in range(nb_eval_steps):
                    eval_action, eval_q, _, _ = agent.step(eval_obs, apply_noise=False, compute_Q=True)
                    eval_obs, eval_r, eval_done, eval_info = eval_env.step(
                        max_action_n * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    if render_eval:
                        eval_env.render()
                    eval_episode_reward += eval_r

                    eval_qs.append(eval_q)
                    for d in range(len(eval_done)):
                        if eval_done[d]:
                            eval_episode_rewards.append(eval_episode_reward[d])
                            eval_episode_rewards_history.append(eval_episode_reward[d])
                            eval_episode_reward[d] = 0.0

        if MPI is not None:
            mpi_size = MPI.COMM_WORLD.Get_size()
        else:
            mpi_size = 1

        # Log stats.
        # XXX shouldn't call np.mean on variable length lists
        duration = time.time() - start_time

        # todo not record agent stats here
        # stats = agent.get_stats()
        # combined_stats = stats.copy()
        # todo simplified log
        combined_stats = {}
        combined_stats['rollout/return'] = np.array(
            [np.mean(epoch_episode_rewards[0]), np.mean(epoch_episode_rewards[1])])
        combined_stats['rollout/return_std'] = np.array(
            [np.std(epoch_episode_rewards[0]), np.std(epoch_episode_rewards[1])])
        combined_stats['rollout/return_history'] = np.array(
            [np.mean(episode_rewards_history[0]), np.mean(episode_rewards_history[1])])
        combined_stats['rollout/return_history_std'] = np.array(
            [np.std(episode_rewards_history[0]), np.std(episode_rewards_history[1])])
        combined_stats['rollout/episode_steps'] = np.array(
            [np.mean(epoch_episode_steps[0]), np.mean(epoch_episode_steps[1])])
        combined_stats['rollout/Q_mean'] = np.array(
            [np.mean(epoch_qs[0]), np.mean(epoch_qs[1])])
        combined_stats['total/duration'] = duration
        combined_stats['total/steps_per_second'] = float(t) / float(duration)
        combined_stats['total/episodes'] = episodes
        combined_stats['rollout/episodes'] = epoch_episodes
        combined_stats['rollout/actions_mean'] = np.array(
            [np.mean(epoch_actions[0]), np.mean(epoch_actions[1])])
        # Evaluation statistics.
        if eval_env is not None:
            combined_stats['eval/return'] = eval_episode_rewards
            combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
            combined_stats['eval/Q'] = eval_qs
            combined_stats['eval/episodes'] = len(eval_episode_rewards)

        def as_scalar(x):
            if isinstance(x, np.ndarray):
                assert x.size == 1
                return x[0]
            elif np.isscalar(x):
                return x
            else:
                raise ValueError('expected scalar, got %s' % x)

        # todo now only log mean reward
        combined_stats_sums = np.array([np.array(x).flatten().mean() for x in combined_stats.values()])
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

    return agent
