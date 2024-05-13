import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np
import math

# Computer vision
import cv2

class CustomDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomDrivingEnv, self).__init__()
        
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(9)  # Define based on your specific needs
        
        # Example for using image as input:
        #self.observation_space = spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)
        # Example for using grayscale image as input:
        self.observation_space = spaces.Box(low=0, high=255, shape=(240, 320), dtype=np.uint8)  # Reduced and grayscale

        # Initialize PyBullet
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.track_width = 1
        self.major_axis = 10
        self.minor_axis = 8
        self.num_cones = 20

    def reset(self):
        # Reset the simulation to initial state
        p.resetSimulation()
        p.loadURDF("plane.urdf")

        carStartPos = [0, 7.5, 0.1]
        carStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.carId = p.loadURDF("racecar/racecar.urdf", carStartPos, carStartOrientation)

        # Create the track
        self.create_oval_track()

        # Reset the state of the environment to an initial state
        # Here, you could define an initial observation
        observation = self.get_observation()

        return observation

    def step(self, action):
        # Execute one time step within the environment
        # Apply action, simulate physics, retrieve new state, reward, check if done
        self.apply_action(action)
        p.stepSimulation()

        observation = self.get_observation()
        reward = self.calculate_reward()
        done = self.check_if_done()

        return observation, reward, done, {}

    def render(self, mode='human'):
        # Render the environment to the screen
        # Optional if you handle rendering in PyBullet GUI
        pass

    def close(self):
        p.disconnect()

    def create_oval_track(self):
        center = (0, 0)
        for i in range(self.num_cones):
            angle = (2 * math.pi * i) / self.num_cones
            outer_x = center[0] + self.major_axis * math.cos(angle)
            outer_y = center[1] + self.minor_axis * math.sin(angle)
            inner_x = center[0] + (self.major_axis - self.track_width) * math.cos(angle)
            inner_y = center[1] + (self.minor_axis - self.track_width) * math.sin(angle)

            p.loadURDF("red_cone 1.urdf", [outer_x, outer_y, 0.2], p.getQuaternionFromEuler([0, 0, 1]))
            p.loadURDF("blue_cone 1.urdf", [inner_x, inner_y, 0.2], p.getQuaternionFromEuler([0, 0, 1]))

    # def get_observation(self):
    #     # Ideally return camera image or other sensory data
    #     return np.random.rand(480, 640, 3)  # Placeholder

    def get_observation(self):
        # Assuming 'camera_image' is obtained from the PyBullet simulation
        # camera_image = np.random.rand(480, 640, 3)  # Placeholder for actual camera image retrieval
        # Generate a random image in uint8
        camera_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        # Resize and convert to grayscale
        processed_image = cv2.cvtColor(cv2.resize(camera_image, (320, 240)), cv2.COLOR_RGB2GRAY)
        return processed_image


    def apply_action(self, action):
        # Apply action to the car (e.g., steering and throttle)
        pass

    def calculate_reward(self):
        # Calculate and return the reward for the current state
        return 0.0

    def check_if_done(self):
        # Determine if the episode is done
        return False
