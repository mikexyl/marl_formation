import multiagent.rendering as marend

class Viewer(marend.Viewer):
    def __init__(self, width, height, display=None):
        super(Viewer, self).__init__(width, height, display)


def make_sector(start_ang, end_ang, radius=10, res=30):
    points = []
