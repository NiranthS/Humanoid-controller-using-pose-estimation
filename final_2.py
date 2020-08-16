import tensorflow as tf
import cv2
import time
import argparse
import math
import posenet
import numpy as np 
import pybullet as p
import pybullet_data

file_name = "humanoid.urdf"
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF( "plane.urdf", 0, 0, 0)
robot = p.loadURDF(file_name,[0,0,1])
p.setGravity(0, 0, -10)

constraint1=p.createConstraint(robot,-1,-1,-1,p.JOINT_FIXED,[0,0,0],[0,0,0],[0,0,0.43])
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()
# time.sleep(10)

def get_anglel(pointa,middle_point0,pointb):
	
	point1=np.copy(pointa)
	point2=np.copy(pointb)
	middle_point=np.copy(middle_point0)
	point1[0]=point1[0]-middle_point[0]
	point1[1]=point1[1]-middle_point[1]
	point2[0]=point2[0]-middle_point[0]
	point2[1]=point2[1]-middle_point[1]
	slope_1=point1[1]/point1[0]
	slope_2=point2[1]/point2[0]
	# slope_1=(point1[1]-middle_point[1])/(point1[0]-middle_point[0])
	# slope_2=(point2[1]-middle_point[1])/(point2[0]-middle_point[0])

	case=0
	tan_of_angle=(slope_1-slope_2)/(1+slope_1*slope_2)
	# cos_of_angle=(1+slope_1*slope_2)/np.sqrt((slope_1-slope_2)**2+(1+slope_1*slope_2)**2)
	angle=math.atan(tan_of_angle)
	# angle=math.acos(cos_of_angle)
	angle=angle*180.0/3.141
	
	if (point1[1]-slope_2*point1[0]>0 ) and slope_2 >0:
		case=1
	elif (point1[1]-slope_2*point1[0]>0 ) and slope_2 <0:
		case=-1
	elif slope_2>0:
		case=-1
	else:
		case=1
	if angle<0:
		angle=180+angle
	return angle,case

def get_angler(pointa,middle_point0,pointb):
	
	point1=np.copy(pointa)
	point2=np.copy(pointb)
	middle_point=np.copy(middle_point0)
	point1[0]=point1[0]-middle_point[0]
	point1[1]=point1[1]-middle_point[1]
	point2[0]=point2[0]-middle_point[0]
	point2[1]=point2[1]-middle_point[1]
	slope_1=point1[1]/point1[0]
	slope_2=point2[1]/point2[0]
	# slope_1=(point1[1]-middle_point[1])/(point1[0]-middle_point[0])
	# slope_2=(point2[1]-middle_point[1])/(point2[0]-middle_point[0])

	case=0
	tan_of_angle=(slope_1-slope_2)/(1+slope_1*slope_2)
	# cos_of_angle=(1+slope_1*slope_2)/np.sqrt((slope_1-slope_2)**2+(1+slope_1*slope_2)**2)
	angle=math.atan(tan_of_angle)
	# angle=math.acos(cos_of_angle)
	angle=angle*180.0/3.141
	
	if (point1[1]-slope_2*point1[0]>0 ) and slope_2 <0:
		case=1
	elif (point1[1]-slope_2*point1[0]>0 ) and slope_2 >0:
		case=-1
	elif slope_2<0:
		case=-1
	else:
		case=1
	if angle<0:
		angle=180+angle
	return angle,case



def get_angle2(point1,middle_point,point2):
	tan_t1=(point1[1]-middle_point[1])/(point1[0]-middle_point[0])
	tan_t2=(point2[1]-middle_point[1])/(point2[0]-middle_point[0])
	# theta1=math.atan(tan_t1)
	# theta2=math.atan(tan_t2)
	tan_angle=(tan_t1-tan_t2)/(1+tan_t1*tan_t2)

	# print(theta1*180.0/3.141,"theta1")
	# print(theta2*180.0/3.141,"theta2")
	# angle=np.abs(theta2-theta1)
	angle=math.atan(tan_angle)
	angle=angle*180.0/3.141
	if angle<10:
		angle=180+angle
	return angle

def get_angleh(pointa,middle_point0):
	point1=np.copy(pointa)
	middle_point=np.copy(middle_point0)
	point1[0]=point1[0]-middle_point[0]
	point1[1]=point1[1]-middle_point[1]

	tan_t1=(point1[1])/(point1[0])
	# tan_angle=(tan_t1-tan_t2)/(1+tan_t1*tan_t2)

	# print(theta1*180.0/3.141,"theta1")
	# print(theta2*180.0/3.141,"theta2")
	# angle=np.abs(theta2-theta1)
	angle=math.atan(tan_t1)
	angle=angle*180.0/3.141
	# if angle<0:
	# 	angle=90+angle
	# else :
	# 	angle=90-angle
	return angle

def main():
	with tf.compat.v1.Session() as sess:
		model_cfg, model_outputs = posenet.load_model(args.model, sess)
		output_stride = model_cfg['output_stride']

		# if args.file is not None:
		#     cap = cv2.VideoCapture(args.file)
		# else:
		cap = cv2.VideoCapture(0)
		    # cap = cv2.VideoCapture(args.cam_id)
		# cap.set(3, args.cam_width)
		# cap.set(4, args.cam_height)

		start = time.time()
		frame_count = 0
		while True:
			input_image, display_image, output_scale = posenet.read_cap(
			cap, scale_factor=args.scale_factor, output_stride=output_stride)

			heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
			    model_outputs,
			    feed_dict={'image:0': input_image}
			)

			pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
			    heatmaps_result.squeeze(axis=0),
			    offsets_result.squeeze(axis=0),
			    displacement_fwd_result.squeeze(axis=0),
			    displacement_bwd_result.squeeze(axis=0),
			    output_stride=output_stride,
			    max_pose_detections=10,
			    min_pose_score=0.15)

			keypoint_coords *= output_scale

			# TODO this isn't particularly fast, use GL for drawing and display someday...
			overlay_image = posenet.draw_skel_and_kp(
			    display_image, pose_scores, keypoint_scores, keypoint_coords,
			    min_pose_score=0.15, min_part_score=0.1)
			# print(overlay_image.shape,"shape")
			overlay_image=cv2.resize(overlay_image,(500,480))
			cv2.imshow('posenet', overlay_image)
			frame_count += 1
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			# print(keypoint_coords[0][5],'sholder')
			# print(keypoint_coords[0][7],'elbow')
			# print(keypoint_coords[0][9],'wrist')
			l_shoulder=keypoint_coords[0][5]
			r_shoulder=keypoint_coords[0][6]
			l_elbow=keypoint_coords[0][7]
			r_elbow=keypoint_coords[0][8]
			l_wrist=keypoint_coords[0][9]
			r_wrist=keypoint_coords[0][10]
			l_hip=keypoint_coords[0][11]
			r_hip=keypoint_coords[0][12]
			l_knee=keypoint_coords[0][13]
			r_knee=keypoint_coords[0][14]
			
			angle_l_elbow,c_l_e=get_anglel(l_wrist,l_elbow,l_shoulder)
			
			angle_r_elbow,c_r_e=get_angler(r_wrist,r_elbow,r_shoulder)
			angle_r_shoulder=(get_angle2(r_elbow,r_shoulder,r_hip))
			angle_l_shoulder=(get_angle2(l_elbow,l_shoulder,l_hip))
			angle_l_leg=get_angle2(l_knee,l_hip,l_shoulder)
			angle_r_leg=get_angle2(r_knee,r_hip,r_shoulder)
			angle_h=get_angleh((l_shoulder+r_shoulder)/2.0,(l_hip+r_hip)/2.0)
			# print(angle_h)
			if angle_r_leg>90:
				angle_r_leg=np.abs(180-angle_r_leg)
			if angle_l_leg>90:
				angle_l_leg=np.abs(180-angle_l_leg)
			# print(angle_r_shoulder,"right",c_r_e)
			# print(get_angle(l_shoulder,l_elbow,l_wrist))
			# print(angle_l_elbow,"left",c_l_e)
			l_elbow_position=(angle_l_elbow)*3.141/180
			l_shoulder_position=(90-angle_l_shoulder)*3.141/180
			l_elbow_joint=20
			l_shoulder_joint=18
			r_elbow_joint=24
			r_shoulder_joint=22
			l_leg_upper_joint=5
			r_leg_upper_joint=0
			# print(angle_l_elbow,"angle",c_l_e)
			# print(angle_r_leg,"right leggg")
			if (angle_r_elbow>180):
				angle_r_elbow=180-angle_r_elbow
			r_elbow_position=(angle_r_elbow)*3.141/180
			r_shoulder_position=(90-angle_r_shoulder)*3.141/180
			l_leg_position=angle_l_leg*3.141/180.0*-1
			r_leg_position=angle_r_leg*3.141/180.0
			h_position=angle_h*3.141/180.0

			p.setJointMotorControl2(robot, 17, p.POSITION_CONTROL, targetPosition =1.57,force = 500)
			if c_l_e >=0:
				p.setJointMotorControl2(robot, 19, p.POSITION_CONTROL, targetPosition =-1.57,force = 500)
				p.setJointMotorControl2(robot, l_elbow_joint, p.POSITION_CONTROL, targetPosition =-1*l_elbow_position,force = 5000)
			else:
				p.setJointMotorControl2(robot, 19, p.POSITION_CONTROL, targetPosition =1.57,force = 500)
				p.setJointMotorControl2(robot, l_elbow_joint, p.POSITION_CONTROL, targetPosition =-1*(np.pi-l_elbow_position),force = 5000)

			
			p.setJointMotorControl2(robot, l_shoulder_joint, p.POSITION_CONTROL, targetPosition =l_shoulder_position,force = 5000)

			if c_r_e <=0:
				p.setJointMotorControl2(robot, 23, p.POSITION_CONTROL, targetPosition =-1.57,force = 500)
				p.setJointMotorControl2(robot, r_elbow_joint, p.POSITION_CONTROL, targetPosition =(np.pi-r_elbow_position),force = 500)
				
			else:
				p.setJointMotorControl2(robot, 23, p.POSITION_CONTROL, targetPosition =1.57,force = 500)
				p.setJointMotorControl2(robot, r_elbow_joint, p.POSITION_CONTROL, targetPosition =r_elbow_position,force = 500)
				

			p.setJointMotorControl2(robot, 21, p.POSITION_CONTROL, targetPosition =1.57,force = 500)
			p.setJointMotorControl2(robot, 11, p.POSITION_CONTROL, targetPosition =h_position,force = 500)
			# p.setJointMotorControl2(robot, 23, p.POSITION_CONTROL, targetPosition =-1.57,force = 500)
			# p.setJointMotorControl2(robot, r_elbow_joint, p.POSITION_CONTROL, targetPosition =3.141-r_elbow_position,force = 500)
			p.setJointMotorControl2(robot, r_shoulder_joint, p.POSITION_CONTROL, targetPosition =r_shoulder_position,force = 500)
			p.setJointMotorControl2(robot,r_leg_upper_joint,p.POSITION_CONTROL,targetPosition=r_leg_position,force=500)
			p.setJointMotorControl2(robot,l_leg_upper_joint,p.POSITION_CONTROL,targetPosition=l_leg_position,force=500)
			p.stepSimulation()

			# x_coord = int(keypoint_coords[0][10][1])
			# aspect_ratio_of_hand = 0.9
			# ht = 60
			# wd = int(ht*aspect_ratio_of_hand)
			# xt = [x_coord-wd,x_coord+wd]
			# yt = [y_coord-2*ht,y_coord]
			# if(xt[0]>0 and xt[1]<640 and yt[0]>0 and yt[1]<480):
			# 	# print(y_coord,x_coord)
			# 	cv2.imshow('hello', overlay_image[yt[0]:yt[1],xt[0]:xt[1]])

			# # FOR FRAMERATE COUNT
			# if(frame_count%20==0):
			# 	print(input_image.shape, output_scale)
			# 	print('Average FPS: ', frame_count / (time.time() - start))





if __name__ == "__main__":
    main()
