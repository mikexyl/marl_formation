import math

import numpy as np
from multiagent.core import Landmark
from multiagent.scenario import BaseScenario

from multirobot.core import World, Vehicle


def add_to_obs_grid(agent, entity, obs, label):
    observed = False
    entity_pos = entity.state.p_pos - agent.state.p_pos
    entity_polar = cart_to_polar(entity_pos)
    [i, j] = find_grid_id(agent, entity_polar)
    if i is not None and j is not None:
        obs[i, j] = label
        observed = True
        # glog.info([i, j, label])
    return obs, observed


# todo may need a util class for static methods
def cart_to_polar(pos):
    polar = np.zeros(2)
    polar[0] = np.linalg.norm(pos)
    polar[1] = math.atan2(pos[1], pos[0])
    # map output of atan2 to [0,2pi]
    polar[1] = -polar[1] + math.pi if polar[1] < 0 else polar[1]
    return polar


def find_grid_id(agent, entity_polar):
    i = None
    j = None
    if agent.fov.dist[0] < entity_polar[0] < agent.fov.dist[1] and \
            math.fabs(entity_polar[1] - agent.state.p_ang) < agent.fov.ang / 2:
        i = math.floor((entity_polar[0] - agent.fov.dist[0]) / agent.fov.dist_res)
        j = math.floor((entity_polar[1] - (agent.state.p_ang - agent.fov.ang / 2)) / agent.fov.ang_res)
    return i, j


def collision_check(agent, world):
    for entity in world.entities:
        if entity is not agent and distance_entities(agent, entity) <= entity.size + agent.size:
            return True
    return False


def distance_entities(entity1, entity2):
    return np.linalg.norm(entity1.state.p_pos - entity2.state.p_pos)


class Scenario(BaseScenario):
    def __init__(self):
        super(Scenario, self).__init__()

        self.eps_form = 0.4
        self.eps_goal = 0.5

        self.rew_form_ratio = -1
        self.rew_goal_ratio = 1

        self.rew_edge = 0.1
        self.rew_success = 50
        self.rew_collision = -100
        self.rew_penalty = 0.5

    def make_world(self):
        world = World()

        # set any world properties first
        world.dim_c = 2
        num_vehicles = 3
        num_agents = 0
        world.num_agents = num_agents
        world.num_vehicles = num_vehicles
        num_landmarks = 16

        # init formation
        world.form_maintainer.set_num_vehicles(num_vehicles)
        world.form_maintainer.load_sample_formation()

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

        world.vehicles = [Vehicle() for i in range(num_vehicles)]
        for i, vehicle in enumerate(world.vehicles):
            vehicle.name = 'vehicle %d' % i
            vehicle.id = i
            vehicle.collide = False
            vehicle.silent = True
            vehicle.size = 0.15
            if i == 0:
                vehicle.color = np.array([1, 0, 0])
            elif i == 1:
                vehicle.color = np.array([0, 1, 0])
            elif i == 2:
                vehicle.color = np.array([0, 1, 1])
            else:
                vehicle.color = np.random.uniform(0, 1, world.dim_color)

        self.reset_world(world)
        return world

    # todo set a arg to choose random or fixed obstacles
    def reset_world(self, world):
        self.reset_vehicles(world)
        world.form_maintainer.reset()

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
        cf, _ = self.formation_reward(agent, world)
        cs, _ = self.success_reward(agent, world)
        cl, _ = self.collision_reward(agent, world)
        rew_total = 0
        if cf and not cs and not cl:
            return self.rew_edge
        else:
            if cl:
                rew_total += self.rew_collision
            if cs:
                rew_total += self.rew_success
        if not rew_total == 0:
            return rew_total
        else:
            return self.rew_penalty

    def formation_reward(self, agent, world):
        # 0 nothing
        # 1 obstacle
        # 2 agents
        # 3 goal
        # 4~4+n vehicles
        # info("%s, vehicle_obs len: %d" % (agent.name, len(agent.vehicles_obs)))
        if len(agent.vehicles_obs) > 1:
            world.form_maintainer.add_edges(
                [(agent.id, vehicle_obs, distance_entities(agent, world.vehicles[vehicle_obs])) for vehicle_obs in
                 agent.vehicles_obs])
            for i in range(len(agent.vehicles_obs)):
                for j in range(i + 1, len(agent.vehicles_obs)):
                    world.form_maintainer.add_edges(
                        [(agent.vehicles_obs[i], agent.vehicles_obs[j],
                          distance_entities(world.vehicles[agent.vehicles_obs[i]],
                                            world.vehicles[agent.vehicles_obs[j]]))])
        displace, formed = world.form_maintainer.formation_exam()
        if formed:
            return True, self.rew_edge
        else:
            return False, 0

    def appr_reward(self, agent, world):
        return 0

    # todo now it's success if observed
    def success_reward(self, agent, world):
        if agent.goal_obs:
            return True, self.rew_success
        else:
            return False, 0

    def collision_reward(self, agent, world):
        if collision_check(agent, world):
            return True, self.rew_collision
        return False, 0

    def observation(self, agent, world):
        # glog.info("obs of " + agent.name)

        # print(agent.name, agent.state.p_ang)
        agent.goal_obs = False
        agent.vehicles_obs = []
        obs = np.zeros(agent.fov.res)
        # get positions of all entities in this agent's reference frame
        for entity in world.landmarks:
            obs, _ = add_to_obs_grid(agent, entity, obs, 1)
        # communication of all other agents
        for other in world.agents:
            if other is agent:
                continue
            obs, _ = add_to_obs_grid(agent, other, obs, 2)
        # todo observe the goal
        obs, observed = add_to_obs_grid(agent, world.goal_landmark, obs, 3)
        if observed:
            agent.goal_obs = True
        # observe other vehicles
        for i, other in enumerate(world.vehicles):
            if other is agent:
                continue
            obs, observed = add_to_obs_grid(agent, other, obs, agent.id + 4)
            if observed:
                agent.vehicles_obs.append(i)

        return obs.reshape(100)

    def done(self, agent, world):
        # check if out of bounds
        if not (0 <= agent.state.p_pos[0] <= world.size_x and
                0 <= agent.state.p_pos[1] <= world.size_y):
            return True
        # check if succeed
        if agent.goal_obs:
            return True

        # todo collision not check here
        # check if collision
        # return collision_check(agent, world)

        return False
