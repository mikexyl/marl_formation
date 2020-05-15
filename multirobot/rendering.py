import multiagent.rendering as marender
import math

#
# class Viewer(marender.Viewer):
#     def __init__(self, width, height, display=None):
#         super(Viewer, self).__init__(width, height, display)


def make_sector(mid_ang, range_ang, radius=10, res=30, filled=False):
    points = []
    for i in range(res):
        ang = mid_ang - range_ang / 2 + range_ang / res * i
        points.append((math.cos(ang) * radius, math.sin(ang) * radius))
    points.append((0, 0))
    if filled:
        return marender.FilledPolygon(points)
    else:
        return marender.PolyLine(points, True)
