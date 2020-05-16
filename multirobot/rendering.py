import math

import multiagent.rendering as marendering


#
# class Viewer(marender.Viewer):
#     def __init__(self, width, height, display=None):
#         super(Viewer, self).__init__(width, height, display)


def make_fov(mid_ang, fov, res=30):
    gs = [make_sector(mid_ang, fov.ang, fov.dist, res, False)]
    dist_range = fov.dist[1] - fov.dist[0]
    dist_res = dist_range / fov.res[0]
    for rd in range(1, fov.res[0]):
        # todo optimise
        points = []
        for i in range(res + 1):
            ang = mid_ang - fov.ang / 2 + fov.ang / res * i
            points.append(
                (math.cos(ang) * (fov.dist[0] + dist_res * rd), math.sin(ang) * (fov.dist[0] + dist_res * rd)))
        gs.append(marendering.PolyLine(points, False))
    for ra in range(1, fov.res[1] + 1):
        ang = mid_ang - fov.ang / 2 + fov.ang / fov.res[1] * ra
        gs.append(marendering.Line((math.cos(ang) * fov.dist[0], math.sin(ang) * fov.dist[0]),
                                   (math.cos(ang) * fov.dist[1], math.sin(ang) * fov.dist[1])))
    return marendering.Compound(gs)


def make_sector(mid_ang, range_ang, dist, res=30, filled=False):
    points = []
    for i in range(res + 1):
        ang = mid_ang - range_ang / 2 + range_ang / res * i
        points.append((math.cos(ang) * dist[1], math.sin(ang) * dist[1]))
    for i in range(res + 1):
        ang = mid_ang + range_ang / 2 - range_ang / res * i
        points.append((math.cos(ang) * dist[0], math.sin(ang) * dist[0]))
    if filled:
        return marendering.FilledPolygon(points)
    else:
        return marendering.PolyLine(points, True)
