import math

import numpy as np
from glog import info
from multiagent.core import Landmark
from multiagent.scenario import BaseScenario

from multirobot import util
from multirobot.core import World, Vehicle

from ruamel import yaml
import os
class Benchmark(object):
    def __init__(self):
        self.stat_rew_condition = np.zeros(3)

    def print(self):
        if __debug__:
            info('episode reward cond stat: %d, %d, %d' % (self.benchmark.stat_rew_condition[0],
                                                           self.benchmark.stat_rew_condition[1],
                                                           self.benchmark.stat_rew_condition[2]))


class Scenario(BaseScenario):
    def __init__(self):
        super(Scenario, self).__init__()

        self.eps_form = 0.1
        self.eps_goal = 1

        self.rew_form_ratio = -1
        self.rew_goal_ratio = 1

        self.rew_edge = 0.1
        self.rew_success = 50
        self.rew_collision = -100
        self.rew_penalty = -0.1

        self.benchmark = Benchmark()

    def make_world(self):
        world = World()

        # set any world properties first
        world.dim_c = 2
        num_vehicles = 1
        num_agents = 0
        world.num_agents = num_agents
        world.num_vehicles = num_vehicles
        num_landmarks = 50
        res_wall = 30
        num_walls = res_wall * 4

        # init formation
        world.form_maintainer.set_num_vehicles(num_vehicles)
        world.form_maintainer.load_sample_formation()

        world.landmarks = [Landmark() for i in range(num_landmarks + num_walls)]
        for i in range(num_landmarks):
            landmark = world.landmarks[i]
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.1
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
        for i in range(num_landmarks, num_landmarks + num_walls):
            wall = world.landmarks[i]
            wall.name = 'wall %d' % i
            wall.collide = True
            wall.movable = False
            wall.state.p_vel = np.zeros(world.dim_p)
            wall.color = np.array([0, 0, 0])
            wall.size = (world.size_x * 2 + world.size_y * 2) / res_wall / 4

        for n in range(res_wall):
            info(num_landmarks + n)
            world.landmarks[num_landmarks + n].state.p_pos = np.array(
                [-world.size_x, -world.size_y + (2 * world.size_y) / res_wall * n])
        for n in range(res_wall):
            info(num_landmarks + n + res_wall)
            world.landmarks[num_landmarks + n + res_wall].state.p_pos = np.array(
                [-world.size_x + (2 * world.size_x) / res_wall * n, -world.size_y])
        for n in range(res_wall):
            info(num_landmarks + n + res_wall * 2)
            world.landmarks[num_landmarks + n + res_wall * 2].state.p_pos = np.array(
                [-world.size_x + (2 * world.size_x) / res_wall * n, world.size_y])
        for n in range(res_wall):
            info(num_landmarks + n + res_wall * 3)
            world.landmarks[num_landmarks + n + res_wall * 3].state.p_pos = np.array(
                [world.size_x, -world.size_y + (2 * world.size_y) / res_wall * n])

        world.goal_landmark = Landmark()
        world.goal_landmark.name = 'goal landmark'
        # world.goal_landmark.state.p_pos = np.array([world.size_x - 1, world.size_x - 1])
        # world.goal_landmark.state.p_pos = np.array([-3.5, -1])
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

    def save(self,world):
        P_posdict = dict()
        for i in range(0,50):
            x_pos = float(world.entities[i].state.p_pos[0])
            y_pos = float(world.entities[i].state.p_pos[1])
            P_posdict['landmark_'+str(i)] = [x_pos,y_pos]
        curpath = os.path.dirname(os.path.realpath(__file__))
        yamlpath = os.path.join(curpath, "scenario_P_pos.yaml")
        with open(yamlpath, "w", encoding="utf-8") as f:
            yaml.dump(P_posdict, f )
        # print(P_posdict['landmark_35'])
        # print(P_posdict)

    def load(self,file_path,world):
        P_posdict = yaml.safe_load(open(file_path, 'r'))
        # print(type(P_posdict))
        # print(P_posdict)
        for j in range(0,50):
            world.entities[j].state.p_pos = a = P_posdict['landmark_'+str(j)]



    # todo set a arg to choose random or fixed obstacles
    def reset_world(self, world):
        self.reset_vehicles(world)
        self.reset_goal(world)
        world.form_maintainer.reset()
        self.benchmark.stat_rew_condition = np.zeros(3)

    @staticmethod
    def reset_goal(world):
        # world.goal_landmark.state.p_pos = np.random.uniform(-1, world.size_x - 1, world.dim_p)
        world.goal_landmark.state.p_pos = np.array([1, 1])

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
        cs, rew_success = self.success_reward(agent, world)
        cl, _ = self.collision_reward(agent, world)
        ce, _ = self.exploration_reward(agent, world)
        self.benchmark.stat_rew_condition += [cf, cs, cl]

        rew_total = 0
        if cf and not cs and not cl:
            return self.rew_edge
        else:
            if cl:
                rew_total += self.rew_collision
            if cs:
                rew_total += rew_success
        if not rew_total == 0:
            return rew_total
        else:
            return self.rew_penalty

    def formation_reward(self, agent, world):
        if not len(world.vehicles) > 1:
            return False, 0
        # 0 nothing
        # 1 obstacle
        # 2 agents
        # 3 goal
        # 4~4+n vehicles
        # info("%s, vehicle_obs len: %d" % (agent.name, len(agent.vehicles_obs)))
        # todo obs changed, formation not changed yet
        if len(agent.vehicles_obs) > 1:
            world.form_maintainer.add_edges(
                [(agent.id, vehicle_obs, util.distance_entities(agent, world.vehicles[vehicle_obs])) for vehicle_obs in
                 agent.vehicles_obs])
            for i in range(len(agent.vehicles_obs)):
                for j in range(i + 1, len(agent.vehicles_obs)):
                    world.form_maintainer.add_edges(
                        [(agent.vehicles_obs[i], agent.vehicles_obs[j],
                          util.distance_entities(world.vehicles[agent.vehicles_obs[i]],
                                                 world.vehicles[agent.vehicles_obs[j]]))])
        displace, formed = world.form_maintainer.formation_exam(self.eps_form)
        if formed:
            return True, self.rew_edge
        else:
            return False, 0

    def appr_reward(self, agent, world):
        return 0

    # todo now it's success if observed
    def success_reward(self, agent, world):
        # if agent.goal_obs:
        #     rew = (agent.dist_to_goal - self.eps_goal) / (agent.fov.dist[1] - self.eps_goal) * self.rew_success
        #     return True, rew
        # else:
        #     return False, 0
        return True, (1 - (agent.dist_to_goal - self.eps_goal) / (
                2 * world.size_x - self.eps_goal)) * self.rew_success * 0.05

    def collision_reward(self, agent, world):
        if util.collision_check(agent, world) or agent.is_stuck:
            return True, self.rew_collision
        return False, 0

    def exploration_reward(self, agent, world):
        # if not agent.is_stuck:
        return False, 0

    def observation(self, agent, world):
        # glog.info("obs of " + agent.name)

        # print(agent.name, agent.state.p_ang)
        agent.goal_obs = False
        agent.vehicles_obs = []

        # obs = np.zeros(agent.fov.res, dtype=np.uint8)
        # for i in range(agent.fov.grid.shape[0]):
        #     for j in range(agent.fov.grid.shape[1]):
        #         cart = util.polar_to_cart(agent.fov.grid[i][j], agent.state.p_ang)
        #         # todo a little to iterate all objects here
        #         for entity in world.entities:
        #             cand = []
        #             if np.linalg.norm(cart - entity.state.p_pos) <= entity.size + agent.size:
        #                 if 'landmark' in entity.name or 'wall' in entity.name:
        #                     cand.append(1)
        #                 elif 'goal' in entity.name:
        #                     cand.append(100)
        #                 elif 'agent' in entity.name:
        #                     cand.append(1)
        #                 elif 'vehicle' in entity.name:
        #                     cand.append(200 + entity.id)
        #             if len(cand)>1:
        #                 obs[i][j] = max(cand)

        # # get positions of all entities in this agent's reference frame
        # for entity in world.landmarks:
        #     obs, _ = util.add_to_obs_grid(agent, entity, obs, 1)
        # # communication of all other agents
        # for other in world.agents:
        #     if other is agent:
        #         continue
        #     obs, _ = util.add_to_obs_grid(agent, other, obs, 2)
        # # todo observe the goal
        # obs, observed = util.add_to_obs_grid(agent, world.goal_landmark, obs, 3)
        # if observed:
        #     agent.goal_obs = True
        # # observe other vehicles
        # for i, other in enumerate(world.vehicles):
        #     if other is agent:
        #         continue
        #     obs, observed = util.add_to_obs_grid(agent, other, obs, agent.id + 4)
        #     if observed:
        #         agent.vehicles_obs.append(i)

        # entity_pos = world.goal_landmark.state.p_pos - agent.state.p_pos
        # entity_polar = util.cart_to_polar(entity_pos)
        # agent.dist_to_goal = entity_polar[0]
        # agent.goal_obs = True
        # # return np.append(obs.reshape(obs.shape[0] * obs.shape[1]), entity_polar)
        # return obs.reshape(obs.shape[0] * obs.shape[1])

        # change obs mode
        obs = np.zeros((len(world.entities), 2))
        for i, entity in enumerate(world.entities):
            if entity is agent:
                obs[i] = np.array([0, 0])
            else:
                entity_pos = entity.state.p_pos - agent.state.p_pos
                entity_polar = util.cart_to_polar(entity_pos)
                if util.in_fov_check(agent, entity_polar):
                    obs[i] = entity_polar
                    if entity is world.goal_landmark:
                        agent.goal_obs = True
                    elif isinstance(entity, Vehicle):
                        agent.vehicles_obs.append(entity.id)
                else:
                    obs[i] = np.array([0, 0])
                if entity is world.goal_landmark:
                    obs[i] = entity_polar
                    agent.dist_to_goal = entity_polar[0]

        return obs.reshape(len(world.entities) * 2)



    def done(self, agent, world):
        # check if succeed
        # if agent.goal_obs:
        #     return True
        # return False
        return agent.is_stuck
        # return util.collision_check(agent, world)
        # todo collision not check here
        # check if collision
        # return util.collision_check(agent, world)

    def benchmark_data(self, agent, world):
        return 0

