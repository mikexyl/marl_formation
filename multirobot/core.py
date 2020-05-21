import math

import multiagent.core as macore
import numpy as np

from formation.maintainer import Maintainer
from multirobot import util


class VehicleState(macore.AgentState):
    def __init__(self):
        super(VehicleState, self).__init__()
        self.p_ang = None


class VehicleAction(macore.Action):
    def __init__(self):
        super(VehicleAction, self).__init__()


class VehicleBodySpeed(object):
    def __init__(self):
        self.p_vel_lin = None
        self.p_vel_ang = None


class FovParams(object):
    def __init__(self, ang=2 / 3 * math.pi, dist=np.array([0.5, 2.5]), res=np.array([10, 10])):
        self.dist = dist
        self.ang = ang
        # res[0] -> res of dist, res[1]-> res of ang
        self.res = res
        self.dist_res = (dist[1] - dist[0]) / res[0]
        self.ang_res = ang / res[1]


class Vehicle(macore.Agent):
    def __init__(self):
        super(Vehicle, self).__init__()
        self.id = 0

        self.max_vel_x = 1
        self.max_vel_y = 1
        self.max_vel_ang = math.pi
        self.state = VehicleState()
        self.action = VehicleAction()
        self.body_speed = VehicleBodySpeed()
        self.size = 0.5
        self.obs = None
        self.accel = 1

        self.fov = FovParams()
        self.vehicles_obs = None
        self.goal_obs = False

        self.is_stuck = False

    # a minimised vehicle model
    # todo a better model
    def premove(self, dt):
        t_ang = None
        t_pos = None
        t_vel = None
        if self.movable:
            self.state.p_vel = np.zeros(2)
            t_ang = self.state.p_ang + (self.body_speed.p_vel_ang * dt)
            t_ang %= (math.pi * 2)
            t_vel = self.body_speed.p_vel_lin \
                .dot(np.array([(math.cos(self.state.p_ang), math.sin(self.state.p_ang)),
                               (math.sin(self.state.p_ang), math.cos(self.state.p_ang))]))
            t_pos = self.state.p_pos + t_vel * dt
        return t_ang, t_pos, t_vel

    def move_to(self, t_ang=None, t_pos=None, t_vel=None):
        if t_vel is not None:
            if t_ang is not None:
                self.state.p_ang = t_ang
            if t_pos is not None:
                self.state.p_pos = t_pos
            if t_pos is not None and t_ang is not None:
                self.state.p_vel = t_vel

    def action_to_speed(self):
        self.body_speed.p_vel_lin = np.array([(self.action.u[0] + 1) * (self.max_vel_x / 2), 0])
        self.body_speed.p_vel_ang = self.action.u[1] * self.max_vel_ang


class World(macore.World):
    def __init__(self):
        super(World, self).__init__()
        self.vehicles = []
        self.goal_landmark = []
        # square world now, only size x is used
        # todo world size should be a gym.spaces.Box?
        self.size_x = 5
        self.size_y = 5
        self.num_agents = 0
        self.num_vehicles = 0
        self.centroid = np.array([-3.5, -3.5])
        self.radius = 1

        # formation init
        # todo put it here since it need to reset when step
        self.form_maintainer = Maintainer()

    # override, return all vehicles
    @property
    def policy_agents(self):
        return [vehicle for vehicle in self.vehicles if vehicle.action_callback is None]

    # override, return all agents, landmarks, goal_landmark and vehicles
    @property
    def entities(self):
        return super(World, self).entities + [self.goal_landmark] + self.vehicles

    # entities except vehicles
    @property
    def env_entities(self):
        return super(World, self).entities + [self.goal_landmark]

    def step(self):
        # super(World, self).step()

        # move vehicles
        self.apply_action_speed()
        for i, vehicle in enumerate(self.vehicles):
            self.update_agent_state(vehicle)

        self.form_maintainer.reset()

    # apply body speed actions
    def apply_action_speed(self):
        for i, vehicle in enumerate(self.vehicles):
            vehicle.action_to_speed()
            t_ang, t_pos, t_vel = vehicle.premove(self.dt)
            if not util.collision_check(vehicle, self, t_pos, vehicle.size):
                vehicle.move_to(t_ang, t_pos, t_vel)
                vehicle.is_stuck = False
            else:
                vehicle.move_to(t_ang, t_vel=t_vel)
                vehicle.is_stuck = True
