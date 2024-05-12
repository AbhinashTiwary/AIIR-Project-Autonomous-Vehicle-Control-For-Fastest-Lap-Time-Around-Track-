import numpy as np
import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


###################################
# # Hyper parameters that will be used in the DQN algorithm

# np.random.seed(0)
# torch.manual_seed(0)
# episode_batch_score = 0
# episode_reward = 0
# np.bool = np.bool_
# #EPISODES = 500                 # number of episodes to run the training for
LEARNING_RATE = 0.00025         # the learning rate for optimising the neural network weights
MEM_SIZE = 10000                 # maximum size of the replay memory - will start overwritting values once this is exceed
REPLAY_START_SIZE = 2000        # The amount of samples to fill the replay memory with before we start learning
BATCH_SIZE = 32                 # Number of random samples from the replay memory we use for training each iteration
GAMMA = 0.99                    # Discount factor
EPS_START = 0.1                 # Initial epsilon value for epsilon greedy action sampling
EPS_END = 0.0001                # Final epsilon value
EPS_DECAY = 4 * MEM_SIZE        # Amount of samples we decay epsilon over
MEM_RETAIN = 0.1                # Percentage of initial samples in replay memory to keep - for catastrophic forgetting
NETWORK_UPDATE_ITERS = 5000     # Number of samples 'C' for slowly updating the target network \hat{Q}'s weights with the policy network Q's weights

FC1_DIMS = 128                   # Number of neurons in our MLP's first hidden layer
FC2_DIMS = 128                   # Number of neurons in our MLP's second hidden layer

# for creating the policy and target networks - same architecture
class Network(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        self.input_shape = env.observation_space.shape
        self.action_space = env.action_space.n

        # build an MLP with 2 hidden layers
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(*self.input_shape, FC1_DIMS),   # input layer
            torch.nn.ReLU(),     # this is called an activation function
            torch.nn.Linear(FC1_DIMS, FC2_DIMS),    # hidden layer
            torch.nn.ReLU(),     # this is called an activation function
            torch.nn.Linear(FC2_DIMS, self.action_space)    # output layer
            )

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()  # loss function

    def forward(self, x):
        return self.layers(x)

# handles the storing and retrival of sampled experiences
class ReplayBuffer:
    def __init__(self, env):
        self.mem_count = 0
        self.states = np.zeros((MEM_SIZE, *env.observation_space.shape),dtype=np.float32)
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, *env.observation_space.shape),dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=bool)

    def add(self, state, action, reward, state_, done):
        # if memory count is higher than the max memory size then overwrite previous values
        if self.mem_count < MEM_SIZE:
            mem_index = self.mem_count
        else:
            ############ avoid catastropic forgetting - retain initial 10% of the replay buffer ##############
            mem_index = int(self.mem_count*.1)
            ##################################################################################################

        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] =  1 - done

        self.mem_count += 1

    # returns random samples from the replay buffer, number is equal to BATCH_SIZE
    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True)

        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        return states, actions, rewards, states_, dones

class DQN_Solver:
    def __init__(self, env):
        self.memory = ReplayBuffer(env)
        self.policy_network = Network(env)  # Q
        self.target_network = Network(env)  # \hat{Q}
        self.target_network.load_state_dict(self.policy_network.state_dict())  # initially set weights of Q to \hat{Q}
        self.learn_count = 0    # keep track of the number of iterations we have learnt for

    # epsilon greedy
    def choose_action(self, observation):
        # only start decaying epsilon once we actually start learning, i.e. once the replay memory has REPLAY_START_SIZE
        if self.memory.mem_count > REPLAY_START_SIZE:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * self.learn_count / EPS_DECAY)
        else:
            eps_threshold = 1.0
        # if we rolled a value lower than epsilon sample a random action
        if random.random() < eps_threshold:
            return np.random.choice(np.array(range(9)), p=[0.11, 0.165, 0.11, 0.11, 0.01, 0.11, 0.11, 0.165, 0.11])  #, p=[0.9,0.1]  # prioritise moving forward or back and less not doing anything

        # otherwise policy network, Q, chooses action with highest estimated Q-value so far
        state = torch.tensor(observation).float().detach()
        state = state.unsqueeze(0)
        self.policy_network.eval()  # only need forward pass
        with torch.no_grad():       # so we don't compute gradients - save memory and computation
            ################ retrieve q-values from policy network, Q ################################
            q_values = self.policy_network(state)
            ##########################################################################################
        return torch.argmax(q_values).item()

    # main training loop
    def learn(self):
        states, actions, rewards, states_, dones = self.memory.sample()  # retrieve random batch of samples from replay memory
        states = torch.tensor(states , dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        states_ = torch.tensor(states_, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)

        self.policy_network.train(True)
        q_values = self.policy_network(states)                # get current q-value estimates (all actions) from policy network, Q
        q_values = q_values[batch_indices, actions]           # q values for sampled actions only

        self.target_network.eval()                            # only need forward pass
        with torch.no_grad():                                 # so we don't compute gradients - save memory and computation
            ###### get q-values of states_ from target network, \hat{q}, for computation of the target q-values ###############
            q_values_next = self.target_network(states_)
            # q_values_next = q_values_next[batch_indices]
            ###################################################################################################################

        # print(q_values_next.shape)
        # print(torch.max(q_values_next, dim=1).shape)
        q_values_next_max = torch.max(q_values_next, dim=1)[0]  # max q values for next state

        q_target = rewards + GAMMA * q_values_next_max * dones  # our target q-value

        ###### compute loss between target (from target network, \hat{Q}) and estimated q-values (from policy network, Q) #########
        loss = self.policy_network.loss(q_target, q_values)
        ###########################################################################################################################

        #compute gradients and update policy network Q weights
        self.policy_network.optimizer.zero_grad()
        loss.backward()
        self.policy_network.optimizer.step()
        self.learn_count += 1

        # set target network \hat{Q}'s weights to policy network Q's weights every C steps
        if  self.learn_count % NETWORK_UPDATE_ITERS == NETWORK_UPDATE_ITERS - 1:
            print("updating target network")
            self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def returning_epsilon(self):
        return self.exploration_rate
