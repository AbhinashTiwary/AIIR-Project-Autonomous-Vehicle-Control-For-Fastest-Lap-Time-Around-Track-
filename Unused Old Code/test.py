import pybullet as p
import os

# Connect to PyBullet
physicsClient = p.connect(p.GUI)  # Use p.DIRECT to run without GUI

print("Current Directory:", os.getcwd())
mesh_path = "red_cone.urdf"
print("Attempting to load:", mesh_path)

try:
    # Load the URDF
    obj_id = p.loadURDF(mesh_path, useFixedBase=True)
    print("URDF loaded successfully.")
except Exception as e:
    print("Failed to load URDF:", str(e))

# Disconnect from PyBullet
p.disconnect()
