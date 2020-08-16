import pybullet as p
import cv2
import pybullet_data
import os
import time
import math
import numpy as np

file_name = "humanoid.urdf"
file2="humanoid/humanoid.urdf"
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF( "plane.urdf", 0, 0, 0)
robot = p.loadURDF(file_name,[0,0,1])
# husky = p.loadURDF("husky/husky.urdf", 0, 0, 0.1)
# quat=p.getQuaternionFromEuler([1.57,0,0])
# p.resetBasePositionAndOrientation(robot, [0, 0, 5],quat)
# cap=cv2.VideoCapture(0)
p.setGravity(0, 0, 0)
print(p.getNumJoints(robot))

# p.createConstraint(robot,-1,-1,-1,p.JOINT_FIXED,[0,0,0],[0,0,0],[0,0,1])
while True:
	# p.setJointMotorControl2(robot,22,p.VELOCITY_CONTROL,-0.00001,5)
	p.stepSimulation()
	time.sleep(1./240.0)

	# p.stepSimulation()