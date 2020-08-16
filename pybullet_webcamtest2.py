import pybullet as p
import cv2
import pybullet_data
import os
import time
import math
import numpy as np

file_name = "humanoid.urdf"
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF( "plane.urdf", 0, 0, 0)
robot = p.loadURDF(file_name,[0,0,1])
# husky = p.loadURDF("husky/husky.urdf", 0, 0, 0.1)
# p.resetBasePositionAndOrientation(robot, 0, 0, 1)
cap=cv2.VideoCapture(0)
p.setGravity(0, 0, 0)
targetpos=0
while True:
	_, image=cap.read()
	hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
	red1_l=np.array([75,50,100])
	red1_u=np.array([90,255,255])
	mask=cv2.inRange(hsv,red1_l,red1_u)
	masked_img=cv2.bitwise_and(hsv,hsv,mask=mask)
	cv2.imshow('img',masked_img)
	cnt_green,im2=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	present=0
	
	for c_red in cnt_green:
		area=cv2.contourArea(c_red)
		if area>2:
			present=1
			print("there")
	p.setJointMotorControl2(robot, 17, p.POSITION_CONTROL, targetPosition =1.57,force = 500)
	p.setJointMotorControl2(robot, 19, p.POSITION_CONTROL, targetPosition =-1.57,force = 500)
	joint=5
	if (present==1):
		targetpos=targetpos-0.1
		# for joint in range(2, 6):
		p.setJointMotorControl2(robot, joint, p.POSITION_CONTROL, targetPosition =targetpos,force = 500)
	else:		
		targetVel = 0
		# for joint in range(2, 6):
		p.setJointMotorControl2(robot, joint, p.VELOCITY_CONTROL, targetVelocity =0,force = 500)
	p.stepSimulation()			
			
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()

	# p.stepSimulation()