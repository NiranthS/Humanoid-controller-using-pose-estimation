import tensorflow as tf
import cv2
import time
import argparse

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()


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
			cv2.imshow('posenet', overlay_image)
			frame_count += 1
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			print(keypoint_coords)
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