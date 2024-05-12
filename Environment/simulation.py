import pybullet as p
import pybullet_data
import time
import math

# Start PyBullet in GUI mode
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setRealTimeSimulation(0)

# Load the ground plane
planeId = p.loadURDF("plane.urdf")

# Load a car model
carStartPos = [0, 7.5, 0.1]
carStartOrientation = p.getQuaternionFromEuler([0, 0, math.pi])
carId = p.loadURDF("racecar/racecar.urdf", carStartPos, carStartOrientation)
carPos, carOri = p.getBasePositionAndOrientation(carId)

p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=30, cameraPitch=-10, cameraTargetPosition=carPos)


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

        outer_positions.append((outer_x, outer_y, 0.2))  # Red cones on the outer track
        inner_positions.append((inner_x, inner_y, 0.2))  # Blue cones on the inner track

    return outer_positions, inner_positions

# Define and create the oval track
center = (0, 0)
major_axis = 10  # Length of the track along the major axis
minor_axis = 8  # Length of the track along the minor axis
num_cones = 20  # Total number of cones
track_width = 1  # Uniform width between the inner and outer tracks

outer_positions, inner_positions = create_oval_track(center, major_axis, minor_axis, num_cones, track_width)

# Load cones to form the track
for pos in outer_positions:
    obj_id = p.loadURDF("red_cone.urdf", pos, p.getQuaternionFromEuler([0, 0, 1]))

for pos in inner_positions:
    obj_id = p.loadURDF("blue_cone.urdf", pos, p.getQuaternionFromEuler([0, 0, 1]))

# Physics settings
p.setGravity(0, 0, -9.81)

width, height = 640, 480
fov, aspect, nearplane, farplane = 60, width / height, 0.1, 100


# Simulation loop
for i in range(1000):
    
    width, height = 640, 480
    fov, aspect, nearplane, farplane = 60, width / height, 0.1, 100

    carPos, carOri = p.getBasePositionAndOrientation(carId)
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
    img = p.getCameraImage(width, height, viewMat, projMat, shadow=1, lightDirection=[1, 1, 1], renderer=p.ER_BULLET_HARDWARE_OPENGL)


    p.stepSimulation()
    time.sleep(1/240)  # Time step size


p.disconnect()

