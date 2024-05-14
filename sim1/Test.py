import gym
import simple_driving
# import pybullet_envs
import pybullet as p
import numpy as np
import math
from collections import defaultdict
import pickle
import torch
import random
import DQ

######################### renders image from third person perspective for validating policy ##############################
# env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True, render_mode='tp_camera')
##########################################################################################################################

######################### renders image from onboard camera ###############################################################
# env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True, render_mode='fp_camera')
##########################################################################################################################

######################### if running locally you can just render the environment in pybullet's GUI #######################
env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True)
##########################################################################################################################

agent = DQ.DQN_Solver(env)
agent.policy_network.load_state_dict(torch.load("policy_network.pkl"))
agent.policy_network.eval()
# frames = []
# frames.append(env.render())

state, info = env.reset()

for i in range(200):
    with torch.no_grad():
        q_values = agent.policy_network(torch.tensor(state, dtype=torch.float32))
    action = torch.argmax(q_values).item() # select action with highest predicted q-value
    state, reward, done, info, bunk = env.step(action)
    #print(action)
    # frames.append(env.render())  # if running locally not necessary unless you want to grab onboard camera image
    if done:
        break

env.close()