import pybullet as p
import os


class ConeR:
    def __init__(self, client, base):
        f_name = os.path.join(os.path.dirname(__file__), 'simpleconeR.urdf')
        self.coneR = client.loadURDF(fileName=f_name,
                   basePosition=[base[0], base[1], 0])


