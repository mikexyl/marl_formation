import math

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
            else:
                vehicle.color = np.random.uniform(0, 1, world.dim_color)

        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.3
            while True:
                landmark.state.p_pos = np.random.uniform(0, world.size_x, world.dim_p)
                # if landmark is too close to vehicles, repick pos
                # todo need a new rule to set landmark, in case they are too far
                if not (world.centroid[0] - world.radius - 0.1 < landmark.state.p_pos[0] < world.centroid[
                    0] + world.radius + 0.1 and
                        world.centroid[1] - world.radius - 0.1 < landmark.state.p_pos[1] < world.centroid[
                            1] + world.radius + 0.1):
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

    def reset_vehicles(self, world):
        # start point set as controid at (1.5,1.5)
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

    def fc_reward(self, agent, world, obs=np.zeros((10,10))):
        # 0 nothing
        # 1 obstacle
        # 2 vehicle
        # 3 goal
        return 0

    def observation(self, agent, world):
        # todo need a form of observation
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)
        # communication of all other agents
        other_agent_pos = []
        for other in world.agents:
            if other is agent: continue
            other_agent_pos.append(other.state.p_pos - agent.state.p_pos)
        # observe other vehicles
        other_vehicle_pos = []
        for other in world.vehicles:
            if other is agent: continue
            other_vehicle_pos.append(other.state.p_pos - agent.state.p_pos)

        # todo observe the goal

        # convert to dist and angle
        entity_in_fov = []
        entity_pos = np.concatenate(entity_pos + other_agent_pos + other_vehicle_pos)
        for i in range(0, len(entity_pos) // 2):
            entity_pos_cur = entity_pos[i * 2: i * 2 + 2]
            entity_dist_cur = np.linalg.norm(entity_pos_cur)
            entity_ang_cur = math.atan2(entity_pos_cur[1], entity_pos_cur[0])
            if entity_dist_cur <= agent.fov_dist and \
                    (math.fabs(entity_ang_cur - agent.state.p_ang) <= agent.fov_ang / 2 or
                     math.fabs(entity_ang_cur - agent.state.p_ang) <= agent.fov_ang / 2 + math.pi * 2):
                entity_in_fov.append(np.array([entity_dist_cur, entity_ang_cur]))
            else:
                # todo how to deal with objects unseen
                entity_in_fov.append(np.array([agent.fov_dist, 0]))

        agent.obs = entity_in_fov
        return entity_in_fov

    def done(self, agent, world):
        if np.min(agent.obs) <= agent.size or \
                agent.state.p_pos[0] < 0 or agent.state.p_pos[1] < 0 or \
                agent.state.p_pos[0] > world.size_x or agent.state.p_pos[1] > world.size_y:
            return True
        else:
            return False
