import pybullet as p
import os


class ConeB:
    def __init__(self, client, base):
        f_name = os.path.join(os.path.dirname(__file__), 'simpleconeB.urdf')
        self.coneB = client.loadURDF(fileName=f_name,
                   basePosition=[base[0], base[1], 0])


