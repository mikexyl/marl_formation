import math

import numpy as np
from multiagent.core import Entity


def add_to_obs_grid(agent, entity, obs, label):
    """
    add an entity to the obs grid
    @param agent: vehicle, of whom the fov is
    @param entity: the entity to add into fov
    @param obs: obs grid
    @param label: label of entity type
    @return: result obs grid
    """
    observed = False
    entity_pos = entity.state.p_pos - agent.state.p_pos
    entity_polar = cart_to_polar(entity_pos)
    [i, j] = find_grid_id(agent, entity_polar)
    if i is not None and j is not None:
        obs[i, j] = label
        observed = True
        # glog.info([i, j, label])
    return obs, observed


def cart_to_polar(cart):
    """
    cartesian coordinates to polar
    @param cart: cartesian coordinates
    @return: plar coordinates
    """
    polar = np.zeros(2)
    polar[0] = np.linalg.norm(cart)
    polar[1] = math.atan2(cart[1], cart[0])
    # map output of atan2 to [0,2pi]
    polar[1] = -polar[1] + math.pi if polar[1] < 0 else polar[1]
    return polar


def polar_to_cart(polar, ang):
    """
    polar coordinates to cartesian
    @param cart: cartesian coordinates
    @return: plar coordinates
    """
    polar[1] += ang
    cart = np.array([polar[0] * math.cos(polar[1]), polar[0] * math.sin(polar[1])])
    return cart


def in_fov_check(agent, entity_polar):
    """
    check a polar coordinate if in fov
    @param agent: of whom fov is
    @param entity_polar: polar coordinates
    @return: bool
    """
    # if agent.fov.dist[0] < entity_polar[0] < agent.fov.dist[1] and \
    # return True
    if entity_polar[0] < agent.fov.dist[1] and \
            math.fabs(entity_polar[1] - agent.state.p_ang) < agent.fov.ang / 2:
        return True
    else:
        return False


def find_grid_id(agent, entity_polar):
    """
    find grid id in which an entity is
    @param agent: fov agent
    @param entity_polar: polar coordinate to find
    @return: i,j: grid id
    """
    i = None
    j = None
    if in_fov_check(agent, entity_polar):
        i = math.floor((entity_polar[0] - agent.fov.dist[0]) / agent.fov.dist_res)
        j = math.floor((entity_polar[1] - (agent.state.p_ang - agent.fov.ang / 2)) / agent.fov.ang_res)
    return i, j


def collision_check(agent, world, pos=None, size=None):
    """
    check agent is colliding with any entity in world
    @param agent: agent to check
    @param world: the world agent is in
    @param pos: the pos to check, if pos is none, pos of agent will be used
    @param size: size to check, if none, size of agent will be used
    @return: bool, if collision
    """
    if pos is None:
        pos = agent.state.p_pos
    if size is None:
        size = agent.size

    # if not (-world.size_x <= pos[0] <= world.size_x and
    #         -world.size_y <= pos[1] <= world.size_y):
    #     return True
    for entity in world.entities:
        if entity is not agent and entity.collide and distance_entities(pos, entity) <= entity.size + size:
            return True
    return False


def distance_entities(entity1, entity2):
    """
    compute distance between entities or coordinates
    @param entity1: entity or coordinates
    @param entity2: entity or coordinates
    @return: distance
    """
    if isinstance(entity1, Entity):
        pos1 = entity1.state.p_pos
    elif isinstance(entity1, np.ndarray):
        pos1 = entity1
    else:
        raise NotImplementedError
    if isinstance(entity2, Entity):
        pos2 = entity2.state.p_pos
    elif isinstance(entity2, np.ndarray):
        pos2 = entity2
    else:
        raise NotImplementedError
    return np.linalg.norm(pos1 - pos2)


