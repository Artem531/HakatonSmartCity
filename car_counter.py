
# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from skimage import measure
from imutils import contours

import tensorflow as tf
from PIL import Image
from core import utils

	
IMAGE_H, IMAGE_W = 416, 416
H, W = None, IMAGE_W
#video_path = "./data/demo_data/road.mp4"
#video_path = 0 # use camera
classes = utils.read_coco_names('./data/coco.names')
num_classes = len(classes)


input_tensor, output_tensors = utils.read_pb_return_tensors(tf.get_default_graph(),
                                                            "./checkpoint/yolov3_cpu_nms.pb",
                                                            ["Placeholder:0", "concat_9:0", "mul_6:0"])


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=5,
	help="# of skip frames between detections")
ap.add_argument("-a", "--alternative", type=int, default=0,
	help="# of skip frames between detections")
ap.add_argument("-l", "--stepLine", type=int, default=100,
	help="# difference between to lines")
ap.add_argument("-sh", "--shift", type=int, default=0,
	help="# shift of lines")
args = vars(ap.parse_args())

shift = args["shift"]
stepLine = args["stepLine"]

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])

# initialize the video writer (we'll instantiate later if need be)
writer = None


# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=2, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either left or right
totalFrames = 0
totalRight = 0
totalLeft = 0
totalDown = 0
totalUp = 0

curRight = 0
curLeft = 0
curDown = 0
curUp = 0


history = []
curHistory = []
time = []
curTime = -1
difference = 50

skipFrames = args["skip_frames"]
# 1 - darkmod 0 - lightmod
switchModKey = 0
# start the frames per second throughput estimator
fps = FPS().start()

def grabTheNextFrame(W, H, vs):
	# grab the next frame and handle if we are reading from either
	# VideoCapture or VideoStream
		
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame
	#print(np.shape(frame))
	# if we are viewing a video and we did not grab a frame then we
	# have reached the end of the video
	if args["input"] is not None and frame is None:
		return 0, 0, frame,  0, 0, 0
	# if the frame dimensions are empty, set them
		
	frame = imutils.resize(frame, width=W)
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	image = Image.fromarray(frame)

	img_resized = np.array(image.resize(size=(IMAGE_H, IMAGE_W)), dtype=np.float32)
	img_resized = img_resized / 255.
	return img_resized, image, frame, rgb, H, W

	
def selectCars(detections, scores, labels, frame, image, args, ct, switchModKey, trackers, rgb):
	mask = np.array(labels == 2) + np.array(labels == 3) + np.array(labels == 5)  + np.array(labels == 7)
	if np.sum(mask) != 0:
		skipFrames = 5
		stepLine = args["stepLine"]
		detections = detections[mask]
		scor = scores[mask]
		lab = labels[mask]
		# print detection boxes
		image = utils.draw_boxes(Image.fromarray(frame, "RGB"), detections, scor, lab, classes, [IMAGE_H, IMAGE_W], show=False)
		frame = np.asarray(image)				
		#print(np.shape(frame))
				
		# loop over the detections
		for i in detections:
			#turn lightmod
			if (switchModKey != 0):
				ct.setMaxDistance(50)
				switchModKey = 0
				# construct a dlib rectangle object from the bounding
				# box coordinates and then start the dlib correlation
				# tracker
			tracker = dlib.correlation_tracker()
				
			startX = i[0]
			startY = i[1]
			endX = i[2]
			endY = i[3]

			bbox = np.array([startX, startY, endX, endY])

			# convert_to_original_size
			detection_size, original_size = np.array([IMAGE_H, IMAGE_W]), np.array(Image.fromarray(frame, "RGB").size)
			ratio = original_size / detection_size
			bbox = list((bbox.reshape(2,2) * ratio).reshape(-1))

			startX = bbox[0]
			startY = bbox[1]
			endX = bbox[2]
			endY = bbox[3]
				
			rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
			tracker.start_track(rgb, rect)

			# add the tracker to our list of trackers so we can
			# utilize it during skip frames
			trackers.append(tracker)
		# if doesn't find anything and img is dark try detect light 
	else :
		skipFrames = 2
		#turn darkmod
		if (switchModKey != 1):
			ct.setMaxDistance(100)
			stepLine = args["stepLine"] / 2
			ct.maxDisappeared = 1
			switchModKey = 1
		image, detections, key = lightDetector(np.asarray(image))
		print(detections)
		#cv2.imshow("Image", image)
		frame = image
		# if find headlight try detect
		if (key == 0):
			for (i, c) in enumerate(detections):
				print(i)
				(x, y, w, h) = cv2.boundingRect(c)
				tracker = dlib.correlation_tracker()
				startX = x
				startY = y
				endX = x + w
				endY = y + h
				
				rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
				tracker.start_track(rgb, rect)

				# add the tracker to our list of trackers so we can
				# utilize it during skip frames
				trackers.append(tracker)
	return trackers, frame, image, startX, startY, endX, endY

	
def countedOrNot(x, y, to, totalDown, totalUp, totalLeft, totalRight, curLeft, curRight, curUp, curDown):
	x = [c[0] for c in to.centroids]
	y = [c[1] for c in to.centroids]
	direction = centroid[0] - np.mean(x)
	to.centroids.append(centroid)

	# check to see if the object has been counted or not
	if not to.counted:
		# if object was on the up side and then apeared 
		# on the down side and motion was high 
		# totalUp += 1
		if (args["alternative"] == 1):
			if np.mean(y) > H // 2 - stepLine  + shift and centroid[1] < H // 2 - stepLine  + shift  and ((y[0] - y[-1])**2 > 10 or (x[0] - x[-1])**2 > 10):
				totalUp += 1
				curUp += 1
				to.counted = True
					
			# if object was on the left  side and then apeared 
			# on the right side and motion was high 
			# totalDown += 1
			elif np.mean(y) < H // 2 + stepLine  + shift and centroid[1] > H // 2 + stepLine + shift and ((y[0] - y[-1])**2 > 10 or (x[0] - x[-1])**2 > 10):
				totalDown += 1
				curDown += 1
				to.counted = True
		else:
			if np.mean(x) > W // 2 - stepLine and centroid[0] < W // 2 - stepLine  and ((y[0] - y[-1])**2 > 10 or (x[0] - x[-1])**2 > 10):
				totalLeft += 1
				curLeft += 1
				to.counted = True
					
			# if object was on the Down  side and then apeared 
			# on the Up side and motion was high 
			# totalRight += 1
			elif np.mean(x) < W // 2 + stepLine and centroid[0] > W // 2 + stepLine  and ((y[0] - y[-1])**2 > 10 or (x[0] - x[-1])**2 > 10):
				totalRight += 1
				curRight += 1
				to.counted = True
	return totalDown, totalUp, totalLeft, totalRight, to, curLeft, curRight, curUp, curDown

	
def drawInfo(frame, args, totalDown, totalUp, totalLeft, totalRight):
	if (args["alternative"] == 1):
		info = [
			("Up", totalUp),
			("Down", totalDown),
			("Status", status),
		]
	else:
		info = [
			("Left", totalLeft),
			("Right", totalRight),
			("Status", status),
		]
		

		# loop over the info tuples and draw them on our frame
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, 10 * i + 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
	return frame
# sess for YOLOv3
with tf.Session() as sess:
	# loop over frames from the video stream
	while True:
		# grab the next frame and handle if we are reading from either
		# VideoCapture or VideoStream
		curTime += 1
		img_resized, image, frame, rgb, H, W = grabTheNextFrame(W, H, vs)
		if args["input"] is not None and frame is None:
			break
		# if we are supposed to be writing a video to disk, initialize
		# the writer
		if args["output"] is not None and writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(W, H), True)
				
		# initialize the current status along with our list of bounding
		# box rectangles returned by either (1) our object detector or
		# (2) the correlation trackers
		status = "Waiting"
		rects = []
		
		# check to see if we should run a more computationally expensive
		# object detection method to aid our tracker
		if totalFrames % skipFrames == 0:
			# set the status and initialize our new set of object trackers
			status = "Detecting"
			trackers = []
			detections = []

			# run YOLOv3
			detections, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis = 0)})
			detections, scores, labels = utils.cpu_nms(detections, scores, num_classes, score_thresh=args["confidence"], iou_thresh=args["confidence"])
			print(labels)
			# Select from all detections only cars, trucks ...
			trackers, frame, image, startX, startY, endX, endY = selectCars(detections, scores, labels, frame, image, args, ct, switchModKey, trackers, rgb)	
		# otherwise, we should utilize our object *trackers* rather than
		# object *detectors* to obtain a higher frame processing throughput
		else:
			# loop over the trackers
			for tracker in trackers:
				# set the status of our system to be 'tracking' rather
				# than 'waiting' or 'detecting'
				status = "Tracking"

				# update the tracker and grab the updated position
				tracker.update(rgb)
				pos = tracker.get_position()

				# unpack the position object
				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())

				# add the bounding box coordinates to the rectangles list
				rects.append((startX, startY, endX, endY))

		# draw a vertical lines in the center of the frame -- once an
		# object crosses this line we will determine whether they were
		# moving 'left' or 'right'
		
		if (args["alternative"] == 1):
			cv2.line(frame, (0, H // 2 + stepLine + shift), (W, H // 2 + stepLine + shift), (0, 255, 255), 2)
			cv2.line(frame, (0, H // 2 - stepLine + shift), (W, H // 2 - stepLine + shift), (0, 125, 255), 2)
		else:
			cv2.line(frame, (W // 2 + stepLine, 0), (W // 2 + stepLine, H), (0, 255, 255), 2)
			cv2.line(frame, (W // 2 - stepLine, 0), (W // 2 - stepLine, H), (0, 125, 255), 2)
		
		# use the centroid tracker to associate the (1) old object
		# centroids with (2) the newly computed object centroids
		objects = ct.update(rects)

		# loop over the tracked objects
		for (objectID, centroid) in objects.items():
			# check to see if a trackable object exists for the current
			# object ID
			to = trackableObjects.get(objectID, None)

			# if there is no existing trackable object, create one
			if to is None:
				to = TrackableObject(objectID, centroid)

			# otherwise, there is a trackable object so we can utilize it
			# to determine direction
			else:
				# the difference between the x-coordinate of the *current*
				# centroid and the mean of *previous* centroids will tell
				# us in which direction the object is moving (negative for
				# 'left' and positive for 'right')
				x = [c[0] for c in to.centroids]
				y = [c[1] for c in to.centroids]
				direction = centroid[0] - np.mean(x)
				to.centroids.append(centroid)
				totalDown, totalUp, totalLeft, totalRight, to, curLeft, curRight, curUp, curDown = countedOrNot(x, y, to, totalDown, totalUp, totalLeft, totalRight, curLeft, curRight, curUp, curDown)
				
			# store the trackable object in our dictionary
			trackableObjects[objectID] = to

			# draw both the ID of the object and the centroid of the
			# object on the output frame
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0) )
			cv2.circle(frame, (centroid[0], centroid[1]), 2, (0, 255, 0), -1)

		# construct a tuple of information we will be displaying on the
		# frame
		frame = drawInfo(frame, args, totalDown, totalUp, totalLeft, totalRight)
			# check to see if we should write the frame to disk
		if writer is not None:
			#print(np.shape(frame))
			writer.write(frame)
		
		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		if (curTime % difference == 0):
			time.append(curTime)
			history.append([totalDown, totalUp, totalLeft, totalRight])
			curHistory.append([curDown, curUp, curLeft, curRight])
			curRight = 0
			curLeft = 0
			curDown = 0
			curUp = 0
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		# increment the total number of frames processed thus far and
		# then update the FPS counter
		totalFrames += 1
		fps.update()

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	# check to see if we need to release the video writer pointer
	if writer is not None:
		writer.release()

	# if we are not using a video file, stop the camera video stream
	if not args.get("input", False):
		vs.stop()

	# otherwise, release the video file pointer
	else:
		vs.release()

	# close any open windows
	cv2.destroyAllWindows()
	
	
#draw plots
import matplotlib.pyplot as plt
names = ['totalDown', 'totalUp', 'totalLeft', 'totalRight']

for i in range(len(history[0])):
	plt.figure()
	plt.xlabel('frames')
	plt.ylabel('amount')
	plt.title(names[i])
	tmp = [j[i] for j in history]
	plt.plot(time, tmp, 'ro', time, tmp, 'k')
	plt.show()
	plt.savefig(names[i] + '.png')

	
	
names = ['curDown', 'curUp', 'curLeft', 'curRight']

for i in range(len(curHistory[0])):
	plt.figure()
	plt.xlabel('frames')
	plt.ylabel('amount')
	plt.title(names[i])
	tmp = [j[i] for j in curHistory]
	plt.plot(time, tmp, 'ro', time, tmp, 'k')
	plt.show()
	plt.savefig(names[i] + '.png')
