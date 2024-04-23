# AIIR-Project-Autonomous-Vehicle-Control-For-Path-Optimisation 
Description:  Our project merges artificial intelligence with a simulated racing environment to develop an autonomous vehicle that can learn the most efficient way around any track.  Utilizing a Convolutional Neural Network (CNN) for visual perception and Reinforcement Learning (RL) for path optimization, the vehicle continuously analyzes cone positions to calculate the fastest racing line with accuracy.

Object Detection: A CNN trained from the PyBullet simulation environment will enable robust detection of the track-defining cones.
Path Planning and Optimization: An RL algorithm will guide trajectory optimization seeking the fastest lap times; rewards are based on speed and staying within the cones.
Odometry: Custom algorithms ensure accurate position updating as the car passes through the center of cone pairs.
Simulation Power: The PyBullet environment allows for rapid iteration, testing, and training data generation.
Potential Applications

Autonomous Driving Research: Technologies developed in this project could contribute to advancements in real-world self-driving car perception and navigation.
Optimized Path Planning: The RL-based path optimization approach has potential uses in fields like logistics and robotics.
