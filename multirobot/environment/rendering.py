import math

import multiagent.rendering as marendering
import numpy as np
from glog import warn

try:
    import pyglet
except ImportError:
    pyglet = None
    warn('pyglet import error, dont display or save videos')

try:
    from pyglet.gl import *
except ImportError:
    pyglet.gl = None
    warn('pyglet.gl import error, dont display or save videos')


class Viewer(marendering.Viewer):
    def render(self, return_rgb_array=False):
        glClearColor(1, 1, 1, 1)
        # if return rgb array, no window shows
        # if not return_rgb_array:
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.transform.enable()
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        self.transform.disable()
        arr = None
        if return_rgb_array:
            # self.window.close()
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep='')
            # In https://github.com/openai/gym-http-api/issues/2, we
            # discovered that someone using Xmonad on Arch was having
            # a window of size 598 x 398, though a 600 x 400 window
            # was requested. (Guess Xmonad was preserving a pixel for
            # the boundary.) So we use the buffer height/width rather
            # than the requested one.
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        # if not return_rgb_array:
        self.window.flip()
        self.onetime_geoms = []
        return arr


def make_fov(fov, res=30):
    gs = [make_sector(fov.ang, fov.dist, res, False)]
    dist_range = fov.dist[1] - fov.dist[0]
    dist_res = dist_range / fov.res[0]
    for rd in range(1, fov.res[0]):
        # todo optimise
        points = []
        for i in range(res + 1):
            ang = - fov.ang / 2 + fov.ang / res * i
            points.append(
                (math.cos(ang) * (fov.dist[0] + dist_res * rd), math.sin(ang) * (fov.dist[0] + dist_res * rd)))
        gs.append(marendering.PolyLine(points, False))
    for ra in range(1, fov.res[1] + 1):
        ang = - fov.ang / 2 + fov.ang / fov.res[1] * ra
        gs.append(marendering.Line((math.cos(ang) * fov.dist[0], math.sin(ang) * fov.dist[0]),
                                   (math.cos(ang) * fov.dist[1], math.sin(ang) * fov.dist[1])))
    return marendering.Compound(gs)


def make_sector(range_ang, dist, res=30, filled=False):
    points = []
    for i in range(res + 1):
        ang = - range_ang / 2 + range_ang / res * i
        points.append((math.cos(ang) * dist[1], math.sin(ang) * dist[1]))
    for i in range(res + 1):
        ang = + range_ang / 2 - range_ang / res * i
        points.append((math.cos(ang) * dist[0], math.sin(ang) * dist[0]))
    if filled:
        return marendering.FilledPolygon(points)
    else:
        return marendering.PolyLine(points, True)
