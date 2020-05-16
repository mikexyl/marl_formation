import multiagent.environment as maenv
import numpy as np


# override ma.env, mainly for rendering
class MultiAgentEnv(maenv.MultiAgentEnv):
    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):
        super(MultiAgentEnv, self).__init__(world, reset_callback, reward_callback, observation_callback,
                                            info_callback,
                                            done_callback, shared_viewer)

        # reset discreate action params
        self.discrete_action_space = False
        self.discrete_action_input = False
        self.force_discrete_action = False

        # reset action space
        from gym import spaces
        from multiagent.multi_discrete import MultiDiscrete
        self.action_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,),
                                            dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

    # override to show the entire environment
    def render(self, mode='human'):
        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                # from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700, 700)

            # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            from multiagent import rendering
            from multirobot import rendering as mrrendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # render fov of vehicles
            for vehicle in self.world.vehicles:
                geom = mrrendering.make_fov(vehicle.fov, 30)
                xform = rendering.Transform()
                geom.set_color(*vehicle.color, alpha=0.5)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            # update bounds to center around agent
            if self.shared_viewer:
                self.viewers[i].set_bounds(-self.world.size_x - 1, self.world.size_x + 1, -self.world.size_y - 1,
                                           self.world.size_y + 1)
            else:
                cam_range = 1
                pos = self.agents[i].state.p_pos
                self.viewers[i].set_bounds(pos[0] - cam_range, pos[0] + cam_range, pos[1] - cam_range,
                                           pos[1] + cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)

            # update fov sectors
            e = len(self.world.entities)
            for v, vehicle in enumerate(self.world.vehicles):
                self.render_geoms_xform[e + v].set_translation(*vehicle.state.p_pos)
                self.render_geoms_xform[e + v].set_rotation(vehicle.state.p_ang)

            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array=mode == 'rgb_array'))

        return results

    # todo no need to copy entire
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        from multiagent.multi_discrete import MultiDiscrete
        from gym import spaces
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index + s)])
                index += s
            action = act
        elif isinstance(action_space, spaces.Box):
            # todo this looks terrible
            if action[0] < action_space.low[0]:
                action[0] = action_space.low[0]
            elif action[0] > action_space.high[0]:
                action[0] = action_space.high[0]

            if action[1] < action_space.low[1]:
                action[1] = action_space.low[1]
            elif action[1] > action_space.high[1]:
                action[1] = action_space.high[1]
        else:
            action = [action]

        # todo what's the origin structure of action? how does it be deleted all?
        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1:
                    agent.action.u[0] = -1.0
                if action[0] == 2:
                    agent.action.u[0] = +1.0
                if action[0] == 3:
                    agent.action.u[1] = -1.0
                if action[0] == 4:
                    agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][0]
                    agent.action.u[1] += action[0][1]
                else:
                    agent.action.u = action
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        else:
            action = action[1:]
            # make sure we used all elements of action
        assert len(action) == 0
