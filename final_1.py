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
p.setGravity(0, 0, 0)


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()
time.sleep(3)

def get_angle(point1,middle_point,point2):
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


def main():
	with tf.compat.v1.Session() as sess:
		model_cfg, model_outputs = posenet.load_model(args.model, sess)
		output_stride = model_cfg['output_stride']

		# if args.file is not None:
		#     cap = cv2.VideoCapture(args.file)
		# else:
		cap = cv2.VideoCapture('test_s.mp4')
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
			angle_l_elbow=get_angle(l_shoulder,l_elbow,l_wrist)
			angle_l_shoulder=(get_angle(l_elbow,l_shoulder,l_hip))
			angle_r_elbow=get_angle(r_shoulder,r_elbow,r_wrist)
			angle_r_shoulder=(get_angle(r_elbow,r_shoulder,r_hip))
			angle_l_leg=get_angle(l_knee,l_hip,l_shoulder)
			angle_r_leg=get_angle(r_knee,r_hip,r_shoulder)
			if angle_r_leg>90:
				angle_r_leg=np.abs(180-angle_r_leg)
			if angle_l_leg>90:
				angle_l_leg=np.abs(180-angle_l_leg)
			# print(angle_l_shoulder)
			# print(get_angle(l_shoulder,l_elbow,l_wrist))
			# print(angle_l_elbow)
			l_elbow_position=(180-angle_l_elbow)*3.141/180
			l_shoulder_position=(90-angle_l_shoulder)*3.141/180
			l_elbow_joint=20
			l_shoulder_joint=18
			r_elbow_joint=24
			r_shoulder_joint=22
			l_leg_upper_joint=5
			r_leg_upper_joint=0
			print(angle_l_leg,"left legggg")
			# print(angle_r_leg,"right leggg")
			if (angle_r_elbow>180):
				angle_r_elbow=180-angle_r_elbow
			r_elbow_position=(180-angle_r_elbow)*3.141/180
			r_shoulder_position=(90-angle_r_shoulder)*3.141/180
			l_leg_position=angle_l_leg*3.141/180.0*-1
			r_leg_position=angle_r_leg*3.141/180.0

			p.setJointMotorControl2(robot, 17, p.POSITION_CONTROL, targetPosition =1.57,force = 500)
			p.setJointMotorControl2(robot, 19, p.POSITION_CONTROL, targetPosition =-1.57,force = 500)
			p.setJointMotorControl2(robot, l_elbow_joint, p.POSITION_CONTROL, targetPosition =-1*l_elbow_position,force = 5000)
			p.setJointMotorControl2(robot, l_shoulder_joint, p.POSITION_CONTROL, targetPosition =l_shoulder_position,force = 5000)

			p.setJointMotorControl2(robot, 21, p.POSITION_CONTROL, targetPosition =1.57,force = 500)
			p.setJointMotorControl2(robot, 23, p.POSITION_CONTROL, targetPosition =-1.57,force = 500)
			p.setJointMotorControl2(robot, r_elbow_joint, p.POSITION_CONTROL, targetPosition =3.141-r_elbow_position,force = 5000)
			p.setJointMotorControl2(robot, r_shoulder_joint, p.POSITION_CONTROL, targetPosition =r_shoulder_position,force = 5000)
			p.setJointMotorControl2(robot,r_leg_upper_joint,p.POSITION_CONTROL,targetPosition=r_leg_position,force=5000)
			p.setJointMotorControl2(robot,l_leg_upper_joint,p.POSITION_CONTROL,targetPosition=l_leg_position,force=5000)
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
