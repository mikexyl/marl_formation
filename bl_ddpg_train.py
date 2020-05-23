from multirobot.ddpg.environment import DdpgEnv
from multirobot.environment import make_env
from multirobot.util import parse_args


def train(arglist):
    # Create environment
    env = make_env(arglist.scenario, arglist, arglist.benchmark)
    ddpg_env = DdpgEnv(env)
    from baselines.ddpg.ddpg import learn

    learn(network="mlp",
          env=ddpg_env,
          total_timesteps=400,
          render=False)


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
