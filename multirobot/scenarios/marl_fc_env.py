import math

import glog
import numpy as np
from multiagent.core import Landmark
from multiagent.scenario import BaseScenario

from multirobot.core import World, Vehicle


class Scenario(BaseScenario):

    def make_world(self):
        world = World()

        # set any world properties first
        world.dim_c = 2
        num_vehicles = 3
        num_agents = 0
        world.num_agents = num_agents
        world.num_vehicles = num_vehicles
        num_landmarks = 16

        world.vehicles = [Vehicle() for i in range(num_vehicles)]
        for i, vehicle in enumerate(world.vehicles):
            vehicle.name = 'vehicle %d' % i
            vehicle.collide = False
            vehicle.silent = False
            vehicle.size = 0.15
            if i == 0:
                vehicle.color = np.array([1, 0, 0])
            elif i == 1:
                vehicle.color = np.array([0, 1, 0])
            elif i == 2:
                vehicle.color = np.array([0, 1, 1])
            else:
                vehicle.color = np.random.uniform(0, 1, world.dim_color)

        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.3
            while True:
                landmark.state.p_pos = np.random.uniform(-world.size_x, world.size_x, world.dim_p)
                # if landmark is too close to vehicles, repick pos
                # todo need a new rule to set landmark, in case they are too far
                if not (world.centroid[0] - world.radius - 0.5 < landmark.state.p_pos[0] < world.centroid[
                    0] + world.radius + 0.5 and
                        world.centroid[1] - world.radius - 0.5 < landmark.state.p_pos[1] < world.centroid[
                            1] + world.radius + 0.5):
                    break
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.color = np.array([0, 0, 0])

        world.goal_landmark = Landmark()
        world.goal_landmark.name = 'goal landmark'
        world.goal_landmark.state.p_pos = np.array([world.size_x - 1, world.size_x - 1])
        world.goal_landmark.state.p_vel = np.zeros(world.dim_p)
        world.goal_landmark.collide = False
        world.goal_landmark.movable = False
        world.goal_landmark.size = 0.1
        # set goal landmark
        world.goal_landmark.color = np.array([0, 0, 1])
        for vehicle in world.vehicles:
            vehicle.goal_a = world.goal_landmark

        self.reset_world(world)
        return world

    # todo set a arg to choose random or fixed obstacles
    def reset_world(self, world):
        self.reset_vehicles(world)

    @staticmethod
    def reset_vehicles(world):
        # start point set as centroid at (1.5,1.5)
        # todo make it an option to choose start point
        radius = np.array([0, world.radius])
        alpha = math.pi * 2 / world.num_vehicles
        for i, vehicle in enumerate(world.vehicles):
            vehicle.state.p_pos = world.centroid + radius.dot(
                np.array([(math.cos(alpha * i), math.sin(alpha * i)), (math.sin(alpha * i), math.cos(alpha * i))]))
            vehicle.state.p_vel = np.zeros(world.dim_p)
            vehicle.state.p_ang = math.pi / 2
            vehicle.state.c = np.zeros(world.dim_c)

    def reward(self, agent, world):
        return self.fc_reward(agent, world)

    def fc_reward(self, agent, world, obs=np.zeros((10, 10))):
        # 0 nothing
        # 1 obstacle
        # 2 vehicle
        # 3 goal
        return 0

    @staticmethod
    def cart_to_polar(pos):
        polar = np.zeros(2)
        polar[0] = np.linalg.norm(pos)
        polar[1] = math.atan2(pos[1], pos[0])
        # map output of atan2 to [0,2pi]
        polar[1] = -polar[1] + math.pi if polar[1] < 0 else polar[1]
        return polar

    @staticmethod
    def find_grid_id(agent, entity_polar):
        i = None
        j = None
        if agent.fov.dist[0] < entity_polar[0] < agent.fov.dist[1] and \
                math.fabs(entity_polar[1] - agent.state.p_ang) < agent.fov.ang / 2:
            i = math.floor((entity_polar[0] - agent.fov.dist[0]) / agent.fov.dist_res)
            j = math.floor((entity_polar[1] - (agent.state.p_ang - agent.fov.ang / 2)) / agent.fov.ang_res)
        return i, j

    def add_to_obs_grid(self, agent, entity, obs, label):
        entity_pos = entity.state.p_pos - agent.state.p_pos
        entity_polar = self.cart_to_polar(entity_pos)
        [i, j] = self.find_grid_id(agent, entity_polar)
        if i is not None and j is not None:
            obs[i, j] = label
            # glog.info([i, j, label])
        return obs

    def observation(self, agent, world):

        # glog.info("obs of " + agent.name)

        # print(agent.name, agent.state.p_ang)
        obs = np.zeros(agent.fov.res)
        # get positions of all entities in this agent's reference frame
        for entity in world.landmarks:
            obs = self.add_to_obs_grid(agent, entity, obs, 1)
        # communication of all other agents
        for other in world.agents:
            if other is agent:
                continue
            obs = self.add_to_obs_grid(agent, other, obs, 2)
        # observe other vehicles
        for other in world.vehicles:
            if other is agent:
                continue
            obs = self.add_to_obs_grid(agent, other, obs, 3)
        # todo observe the goal
        obs = self.add_to_obs_grid(agent, world.goal_landmark, obs, 4)

        return obs

    def done(self, agent, world):
        if np.min(agent.obs) <= agent.size or \
                agent.state.p_pos[0] < 0 or agent.state.p_pos[1] < 0 or \
                agent.state.p_pos[0] > world.size_x or agent.state.p_pos[1] > world.size_y:
            return True
        else:
            return False
