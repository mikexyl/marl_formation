## Install

1. Denpendencies:
	- (optional) cuda,cudnn for tensorflow-gpu, if you have a compatible gpu.
	- python >= 3.5, pybullet, and tensorflow(and tensorflow-gpu if available) >= 1.4. make sure python version is compatible with tensorflow
2. Install:
	- it's recommemded to use virtual python environment to manage dependencies, to do so you can install virtualenv and 

		```
		virtualenv <path_to_env> --with-python=python3.6
		```

		to create a virtual environment and specity python version as 3.6 if you have multiple versions of python installed. and then

		```
		source <path_to_env>/bin/activate
		```

		to activate the virtualenv.
	- install cuda and cudnn according to nvidia installation guide if you want to use it.
	- install some dependency python package mentioned before
	
		```
		pip install pybullet tensorflow==1.4 [tensorflow-gpu==1.4]
		```

	- clone repositories [baselines](https://github.com/LXYYY/baselines.git), [maddpg](https://github.com/openai/maddpg.git), [multiagent-particle-envs](https://github.com/LXYYY/multiagent-particle-envs.git)
	- install these package using 

		```
		pip install -e <package_name>
		```

		you can also install baselines from pip remote source by omitting `-e` optioin, but that will ask you to install mujoco instead of pybullet, of which the former asks licenses and more troubling to isntall.
		if pip says there is any dependency missed, just install it with pip.

	- clone this repo, then 

		```
		pip install -e marl_formation
		```

		and try the test steps following. 
		notice: pip install this repo may still leave some dependencies uninstalled, for the setup.py of this package has not been carefully tested.

## Test:

1. run experiments/env_test.py to test if you have installed anything you need, and can load and show the scenario.
	```
	cd marl_formation
	python experiments/env_test.py --scenario=<scenario_name>.py
	```
	this env_test.py only loads and shows the scenario file, and give agents some simple actions, to test if the scenario is built as you thought. here we use it to test your installation.
    
    this scenario script, <scenario_name>.py should be placed in `multirobot/scenarios`. and noticeably, we only need to add `.py`, to input the full name of the script here, but it's not the case in steps below.
2. There are three algorithms in this project: original maddpg, ddpg from openai/baselines and the maddpg built on top of baselines/ddpg, here we annote them as *maddpg*, *bl_ddpg*, *bl_maddpg*. At this stage of project, we should test them with single agent scenario.
	- to train with a scenario,
        ```
        python experiments/<algo>_train.py --scenario=<scenario_name>
        ```
    - to train in any scenario, you need to put `<scenario_name>.py` file in `multirobot/scenarios`. you can find more sample scenarios in source code of `multiagent-particle-envs`, for now, you need to copy the scenario you want to test to `multirobot/scenarios`. theoretically you can just import `multiagent.scenarios` instead of `multirobot.scenarios` to directly load the sample scenario without copying, but I am not sure if this can work without exceptions.
    - the only way to switch in different algorithms to test is to change the scripts to run, the argument `--algorithm` doesn't work now.
    - there are other argument you may need to set. you can use
        ```
        python experiments/<any_script> --help 
        ```
      to see what arguments are available. there are several set of import arguments:
        - training settings. e.g. `--nb_epoch_cycles`, `--nb_epochs` etc. to set epochs, cycles and steps of training. also `--lr`, etc. can let you set learning rates.
        - saving settings. e.g. `--log_path` is to set path of log, where data like rewards will be logged, this arg only works in bl_ddpg. In maddpg, the log saving is not implemented. In bl_maddpg, you need to set `--result_path`, and all results including rewards, model vars, plottings, will be saved in result path. also, only in bl_maddpg, you can use args like `--save_model, --save_actions`, etc. to choose if to save these results or not, and later you can use `--restore` to load model vars from the result path you provided.
        - display setting. e.g. `--display`, you can turn on or off display.
    - besides `marl_fc_env.py`, only `simple.py` scenario is tested workable with three algorithms. theoretically if `simple.py` works, others should just work fine. But if not, you need to do some modification according to the errors you get. notice: unfortunately the environment interfaces wrapping scenarios required by 3 algorithms all differ a little with each other, in data structure, etc. to solve this difference, *maddpg* uses `multiagent.environment`, *bl_ddpg* uses `multirobot.ddpg.environment` and *bl_maddpg* uses `multirbobot.maddpg.environment` to wrap scenarios. So if algorithms can't work on some scenario, you may need have a closer look into these modules.
    - and you need to make sure all multi agent scenarios set to have only one agent, or theoretically bl_ddpg can't run, unless you add some lines to concatenate data from agents.
 

## Change Log
29/05/2020: Tag *bl_maddpg_test_0_1*, bl_maddpg version 0.1, ready to test, 
with following features:

1. indivisual critics, centralized training, decentralized actuation.

with unimplemented:

1. log can't show individual rewards, only mean values
2. shared_critic not implemented
3. all noise and normalization not implemented
4. mpi parallel training
    
24/05/2020: added ddpg from baselines training with the marl_fc_env, got workable result, so maybe something wrong with maddpg
