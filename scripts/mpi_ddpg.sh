#!/usr/bin/env bash

mpiexec -n 4 python -m mpi4py ../bl_ddpg_train.py --scenario=marl_fc_env --benchmark --log_path=/tmp/ddpg_fov_obs_log/