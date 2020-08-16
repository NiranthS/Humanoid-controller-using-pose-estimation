import pybullet as p
import cv2
import pybullet_data
import os
import time
import math
import numpy as np
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

file_name = currentdir + "/humanoid.urdf"
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf", 0, 0, 0)
robot = p.loadURDF(file_name, 0, 0, 0.5)
cap = cv2.VideoCapture(0)
p.setGravity(0, 0, 0)
numJoints = p.getNumJoints(robot)  # 25
action = [i for i in range(numJoints)]

scale = 100.0


def nothing(x):
    pass

print(numJoints,"joints")
cv2.namedWindow("controller with scale" + str(scale),cv2.WINDOW_NORMAL)
# create trackbars for joints
for i in range(0,5):
    cv2.createTrackbar(str(i), "controller with scale" + str(scale), 00, 100, nothing)
# create switch for ON/OFF functionality
switch = "0 : OFF \n1 : ON"
cv2.createTrackbar(switch, "controller with scale" + str(scale), 0, 1, nothing)

while 1:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    actions = [
        cv2.getTrackbarPos(str(i), "controller with scale" + str(scale)) / scale
        for i in range(numJoints)
    ]
    s = cv2.getTrackbarPos(switch, "controller with scale" + str(scale))
    for i in range(numJoints):
        p.setJointMotorControl2(
            robot, i, p.POSITION_CONTROL, targetPosition=actions[i], force=500
        )
    p.stepSimulation()


cv2.destroyAllWindows()
