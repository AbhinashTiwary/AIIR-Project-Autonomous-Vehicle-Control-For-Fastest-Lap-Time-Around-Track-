import pybullet as p
import os
import math


class Car:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'simplecar.urdf')
        self.car = self.client.loadURDF(fileName=f_name,
                              basePosition=[0, 0, 0.1])

        # Joint indices as found by p.getJointInfo()
        self.steering_joints = [0, 2]
        self.drive_joints = [1, 3, 4, 5]
        # Joint speed
        self.joint_speed = 0
        # Drag constants
        self.c_rolling = 0.2
        self.c_drag = 0.01
        # Throttle constant increases "speed" of the car
        self.c_throttle = 100

    def get_ids(self):
        return self.car

    def apply_action(self, action):
        # Expects action to be two dimensional
        throttle, steering_angle = action

        # Clip throttle and steering angle to reasonable values
        throttle = min(max(throttle, -1), 1)
        steering_angle = max(min(steering_angle, 0.6), -0.6)

        # Set the steering joint positions
        self.client.setJointMotorControlArray(self.car, self.steering_joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[steering_angle] * 2)

        # Calculate drag / mechanical resistance ourselves
        # Using velocity control, as torque control requires precise models
        friction = -self.joint_speed * (self.joint_speed * self.c_drag +
                                        self.c_rolling)
        acceleration = self.c_throttle * throttle + friction
        # Each time step is 1/240 of a second
        self.joint_speed = min(self.joint_speed + 0.01 * acceleration, 10.0)
        # if self.joint_speed < 0:
            # self.joint_speed = 0

        # Set the velocity of the wheel joints directly
        self.client.setJointMotorControlArray(
            bodyUniqueId=self.car,
            jointIndices=self.drive_joints,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[self.joint_speed] * 4,
            forces=[1.2] * 4)

    def get_observation(self):
        # Get the position and orientation of the car in the simulation
        pos, ang = self.client.getBasePositionAndOrientation(self.car)
        ang = p.getEulerFromQuaternion(ang)
        ori = (math.cos(ang[2]), math.sin(ang[2]))
        pos = pos[:2]
        # Get the velocity of the car
        vel = self.client.getBaseVelocity(self.car)[0][0:2]

        # Concatenate position, orientation, velocity
        observation = (pos + ori + vel)

        return observation









