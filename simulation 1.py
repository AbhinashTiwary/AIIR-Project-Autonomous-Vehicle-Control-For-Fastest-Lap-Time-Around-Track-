import gym
import simple_driving
import pybullet as p
import pybullet_data
import time
import math
import DQ
import torch

#from pybullet_envs.bullet.racecarGymEnv import RacecarGymEnv

def create_oval_track(center, major_axis, minor_axis, num_cones, track_width):
    """
    Creates positions for an oval track with two lanes.
    - center: (x, y) tuple for the oval's center.
    - major_axis: Length of the major axis for the outer track.
    - minor_axis: Length of the minor axis for the outer track.
    - num_cones: Number of cones along the full track.
    - track_width: Distance between the inner and outer track.
    """
    outer_positions = []
    inner_positions = []

    # Calculate the axes for the inner track
    inner_major_axis = major_axis - track_width
    inner_minor_axis = minor_axis - track_width

    for i in range(num_cones):
        angle = (2 * math.pi * i) / num_cones  # Angle varies from 0 to 2*pi
        outer_x = center[0] + major_axis * math.cos(angle)
        outer_y = center[1] + minor_axis * math.sin(angle)
        inner_x = center[0] + inner_major_axis * math.cos(angle)
        inner_y = center[1] + inner_minor_axis * math.sin(angle)

        outer_positions.append((outer_x, outer_y, 0.1))  # Red cones on the outer track
        inner_positions.append((inner_x, inner_y, 0.1))  # Blue cones on the inner track

    return outer_positions, inner_positions

def make_goal(point_left, point_right):
    x = (point_left[0] + point_right[0])/2
    y = (point_left[1] + point_right[1])/2
    goal_pos = (x, y, 0)
    return goal_pos

def moveCar(action, car_speed):
    throttle, steering_angle = action
    throttle = min(max(throttle, -1), 1)

    steering_angle = max(min(steering_angle, 0.6), -0.6)
    p.setJointMotorControlArray(carId, steering_joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[steering_angle] * 2)
    
    p.setJointMotorControlArray(carId, drive_joints, p.VELOCITY_CONTROL, [throttle + car_speed]*4, [10]*4)
    p.stepSimulation()

def check4goal(pos1, pos2):
    # CNN confirm is cone and colours
    ...
    # if not cone and/or colours return false/None
    return make_goal(pos1, pos2)

def RL_step(action):
    Done = False
    carOd = observation()
    #from possible actions
    fwd = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
    steerings = [-0.6, 0, 0.6, -0.6, 0, 0.6, -0.6, 0, 0.6]
    throttle = fwd[action]
    steering_angle = steerings[action]
    action = [throttle, steering_angle]
    #playout action
    moveCar(action, 1)
    #done = False
    #give updates
    carPos, carOri = p.getBasePositionAndOrientation(carId)
    #calc rewards/goal reached
    dist_to_goal = math.sqrt(((carPos[0] - Car_goal[0]) ** 2 +
                                  (carPos[1] - Car_goal[1]) ** 2))
    reward = -dist_to_goal
    if dist_to_goal < 1:
            print("reached goal")
            reward+=50
            Done = True
            goal_reached = True

    
    
    return carOd, reward, Done

def observation():
    carPos, carOri = p.getBasePositionAndOrientation(carId)
    invCarPos, invCarOrn = p.invertTransform(carPos, carOri)
    #print(invCarPos, invCarOrn, Car_goal, baseOri)
    goalPosInCar, goalOrnInCar = p.multiplyTransforms(invCarPos, invCarOrn, Car_goal, baseOri)
    return [goalPosInCar[0], goalPosInCar[1]]
    
# Start PyBullet in GUI mode
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setRealTimeSimulation(0)
env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True)
# load network
agent = DQ.DQN_Solver(env)
agent.policy_network.load_state_dict(torch.load("policy_network.pkl"))

# Load the ground plane
planeId = p.loadURDF("plane.urdf")
# Load a car model
carStartPos = [0, 7.5, 0.05]
Car_goal = [0,0,0]
carStartOrientation = (0,0,1,0) #p.getQuaternionFromEuler([0, 0, 0])
carId = p.loadURDF("racecar/racecar.urdf", carStartPos, carStartOrientation)
#for i in range (p.getNumJoints(carId)):
	#p.setJointMotorControl2(carId,i,p.POSITION_CONTROL,0)
	#print(p.getJointInfo(carId,i))
steering_joints = [4, 6]
drive_joints = [2, 3, 5, 7]
#car = p.Racecar(physicsClient)
carPos, carOri = p.getBasePositionAndOrientation(carId)

p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=30, cameraPitch=-10, cameraTargetPosition=carPos)

# Define and create the oval track
center = (0, 0)
baseOri = (0,0,0,1)
major_axis = 11  # Length of the track along the major axis
minor_axis = 8  # Length of the track along the minor axis
num_cones = 20  # Total number of cones
track_width = 1  # Uniform width between the inner and outer tracks
current_goal = 6
goal_reached = False 
done = False

outer_positions, inner_positions = create_oval_track(center, major_axis, minor_axis, num_cones, track_width)
cones_pos = [inner_positions, outer_positions]
#for i in range(num_cones):
    #print(make_goal(cones_pos[0][i], cones_pos[1][i]))
# Load cones to form the track
for pos in outer_positions:
    obj_id = p.loadURDF("red_cone 1.urdf", pos, p.getQuaternionFromEuler([0, 0, 1]))

for pos in inner_positions:
    obj_id = p.loadURDF("blue_cone 1.urdf", pos, p.getQuaternionFromEuler([0, 0, 1]))

# Physics settings
p.setGravity(0, 0, -9.81)
#print(p.getBasePositionAndOrientation(carId))
# Simulation loop
state = observation()

width, height = 640, 480
fov, aspect, nearplane, farplane = 60, width / height, 0.1, 100
for i in range(1000):
    Car_goal = check4goal(cones_pos[0][current_goal], cones_pos[1][current_goal])
    done = False
    while (not done):
        with torch.no_grad():
            q_values = agent.policy_network(torch.tensor(state, dtype=torch.float32))
        action = torch.argmax(q_values).item() # select action with highest predicted q-value
        state, reward, done = RL_step(action)
        goal_reached = done
        
        carMat = p.getMatrixFromQuaternion(carOri)
        
        forwardVec = [carMat[0], carMat[3], carMat[6]]
        upVec = [carMat[2], carMat[5], carMat[8]]

        camHeightOffset = 0.1  # Increase this value to raise the camera higher
        camPos = [
        carPos[0] + 0.3 * forwardVec[0], 
        carPos[1] + 0.3 * forwardVec[1], 
        carPos[2] + 0.3 * forwardVec[2] + camHeightOffset
        ]

        
        targetPos = [carPos[0] + 1 * forwardVec[0], carPos[1] + 1 * forwardVec[1], carPos[2] + 1 * forwardVec[2]]
        viewMat = p.computeViewMatrix(camPos, targetPos, upVec)
        projMat = p.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)

        # Get camera image
        #img = p.getCameraImage(width, height, viewMat, projMat, shadow=1, lightDirection=[1, 1, 1], renderer=p.ER_BULLET_HARDWARE_OPENGL)
        #print(goal_reached)
        if goal_reached:
            current_goal+=1
            if current_goal >= 20:
                current_goal = 0
            #print("goal reached")
            break
        time.sleep(1/240)  # Time step size


    # p.stepSimulation()
    # moveCar([1, -0.1])
    

p.disconnect()

