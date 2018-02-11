'''
Special thanks to Adrian Rosebrock from Pyimagesearch
This script is based on his previous work

The script aims to filter Personally Identifiable Information from real-time
camera feeds (either RTSP based IP-Cameras or USB-cameras). The regions of
interests are detected using a DNN. Multiple methods of anonymization can be
used. Such as blurring, gaussian blurring and masking.
The original stream can be saved seperately (optionally with AES encryption)

Please note that this script does not work out of the box. It needs certain
(python) dependencies, folder structures and file-based pipes (mkfifo).

Known limitations: Install argument does not work due to the use of Python
Virtual Environmnents
'''

import argparse

# arguments
ap = argparse.ArgumentParser()
ap.add_argument("-z", "--blur", type=str, default="nothing",
        help="Set to yes to apply blurring to the detections")
ap.add_argument("-Z", "--blurmedian", type=str, default="nothing",
        help="Set to yes to apply median blurring to the detections")
ap.add_argument("-x", "--blurgaussian", type=str, default="nothing",
        help="Set to yes to apply gaussian blurring to the detections")
ap.add_argument("-y", "--warping", type=str, default="nothing",
        help="Set to yes to apply warping to the detections")
ap.add_argument("-Y", "--fill", type=str, default="nothing",
        help="Set to yes to apply filling to the detections")
ap.add_argument("-D", "--draw", required=False, type=str, default="no",
        help="Set to yes to draw a rectangle on the detections")
ap.add_argument("-E", "--encrypt", type=str, default="nothing",
        help="Set to yes to apply AES encryption to the original stream")
ap.add_argument("-b", "--blurlevel", type=int, default=50,
        help="Set the blur intensity (size of pixel square")
ap.add_argument("-B", "--blurpadding", required=False, type=int, default=15,
        help="Set the blur padding applied to the frames")
ap.add_argument("-c", "--codec", type=str, default="X264",
        help="Set the type of codec of output video")
ap.add_argument("-C", "--confidence", type=float, default=0.2,
        help="Confidence level, to filter out weak/incorrect detections")
ap.add_argument("-d", "--detection", type=str, default="no",
        help="Set to yes to turn on detections using DNN")
ap.add_argument("-i", "--feedip", required=False, default=50,
        help="Set the URL to the camera")
ap.add_argument("-e", "--feedusb", required=False, default=50,
        help="Set the USB feed to be used (USB socket number as X, aka /dev/videoX)")
ap.add_argument("-f", "--frames", type=int, default=20,
        help="Set the FPS rate for the video output file")
ap.add_argument("-I", "--install", type=str, default="nothing",
        help="Install opencv on the specified platform: ubuntu-16.04 (Note: this does not work)")
ap.add_argument("-l", "--labels", type=str, default="no",
        help="Set to 'yes' to label the detections")
ap.add_argument("-L", "--logging", type=str, default="nothing",
        help="Specify the optional logfile")
ap.add_argument("-m", "--model", required=False,
        help="path to Caffe pre-trained DNN model")
ap.add_argument("-o", "--original", type=str, default="nothing",
        help="path to output unaltered video file")
ap.add_argument("-O", "--output", type=str, default="nothing",
        help="path to output video file")
ap.add_argument("-R", "--restream", type=str, default="nothing",
        help="restream the edited feed")
ap.add_argument("-p", "--prototxt", required=False,
        help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-s", "--showlocal", type=str, default="no",
        help="Show the video stream on the local machine")
ap.add_argument("-t", "--timer", type=int, required=False, default=9999,
        help="Set the time the script runs")
args = vars(ap.parse_args())

#print used settings
print("[SETTINGS]\n" + \
"    blur: " + str(args["blur"]) + "\n" + \
"    blurmedian: " + str(args["blurmedian"]) + "\n" + \
"    blurgaussian: " + str(args["blurgaussian"]) + "\n" + \
"    warping: " + str(args["warping"]) + "\n" + \
"    fill: " + str(args["fill"]) + "\n" + \
"    encrypt: " + str(args["encrypt"]) + "\n" + \
"    blurlevel: " + str(args["blurlevel"]) + "\n" + \
"    blurpadding: " + str(args["blurpadding"]) + "\n" + \
"    codec: " + str(args["codec"]) + "\n" + \
"    confidence: " + str(args["confidence"]) + "\n" + \
"    detection: " + str(args["detection"]) + "\n" + \
"    draw: " + str(args["draw"]) + "\n" + \
"    feedip: " + str(args["feedip"]) + "\n" + \
"    feedusb: " + str(args["feedusb"]) + "\n" + \
"    frames: " + str(args["frames"]) + "\n" + \
"    install: " + str(args["install"]) + "\n" + \
"    labels: " + str(args["labels"]) + "\n" + \
"    logging: " + str(args["logging"]) + "\n" + \
"    model: " + str(args["model"]) + "\n" + \
"    output: " + str(args["output"]) + "\n" + \
"    original: " + str(args["original"]) + "\n" + \
"    prototxt: " + str(args["prototxt"]) + "\n" + \
"    restream: " + str(args["restream"]) + "\n" + \
"    showlocal: " + str(args["showlocal"]) + "\n" + \
"    timer: " + str(args["timer"]))

# imports
if not args["install"] == "ubuntu-16.04":
	from imutils.video import VideoStream
	from imutils.video import FPS
	from multiprocessing import Process
	from multiprocessing import Queue
	from timeit import default_timer
	#from Crypto.Cipher import AES
	import numpy as np
	import imutils
	import time
	import cv2
	import time
	import os
	import math
	import base64
	import sys
	from subprocess import PIPE, Popen
else:
	import os

if args["encrypt"] == "yes":
	p1 = Popen(['gpg', '--batch', '--yes', '-o', '/tmp/encryptedvideo.gpg', '--passphrase-file', '/tmp/key', '--symmetric', '/tmp/videopipe.avi'], stdout=PIPE)

if args["restream"] == "yes":
	p2 = Popen(['ffserver'], stdout=PIPE)
	p3 = Popen(['ffmpeg', '-re', '-i', '/tmp/videopipe.avi', 'http://localhost:8090/feed2.ffm'], stdout=PIPE)

# install opencv on specified platform (needs work)
if args["install"] == "ubuntu-16.04":
	os.system("sh install/ubuntu-16.04.sh")
	exit()

# frame class
def classify_frame(net, inputQueue, outputQueue):
	# start looping
	while True:
		# check input queue
		if not inputQueue.empty():
			# grab the frame, resize, create blob
			frame = inputQueue.get()
			frame = cv2.resize(frame, (300, 300))
			blob = cv2.dnn.blobFromImage(frame, 0.007843,
				(300, 300), 127.5)

			# set blob as input for detector
			net.setInput(blob)
			detections = net.forward()

			# write detections to output queue
			outputQueue.put(detections)

# checks
BlurPadding = int(args["blurpadding"])
FeedSelector = int(args["feedusb"])
IpFeed = args["feedip"]
if args["feedusb"] == 50:
        FeedSelector = str(args["feedip"])
        args["feedusb"] = "nothing"
#testing = 1

# initialize FourCC, video writer, frame dimensions, and zeros array
fourcc = cv2.VideoWriter_fourcc(*args["codec"])
writer = None
writer2 = None
(h, w) = (None, None)
zeros = None

# MobileNet class labels and label colors
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load serialized model
print("[INFO] loading model " + args["model"] + "...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the input frames, output detections, and the actual detections
inputQueue = Queue(maxsize=1)
outputQueue = Queue(maxsize=1)
detections = None

# independent child process
print("[INFO] starting child process...")
p = Process(target=classify_frame, args=(net, inputQueue,
	outputQueue,))
p.daemon = True
p.start()

# start video stream
print("[INFO] starting input video stream " + str(FeedSelector) + "...")
if args["feedusb"] == 50:
	vs = VideoStream(FeedSelector).start()
else:
	vs = VideoStream(src=FeedSelector).start()

# allow the video stream to start up
time.sleep(2.0)

# initialize timer
start = default_timer()

# start fps counter
fps = FPS().start()

# video frames loop
while True:
	# grab video stream frame, resize it, and grab dimensions
	frame = vs.read()
	frame = imutils.resize(frame, width=600)
	(fH, fW) = frame.shape[:2]
	Original = frame

	# check if the video writer is None
	if writer is None or writer2 is None:
		# initialzie the video writer and store the frame
		(h, w) = frame.shape[:2]
		writer = cv2.VideoWriter(args["original"], fourcc, args["frames"],
			(w * 1, h * 1), True)

		writer2 = cv2.VideoWriter(args["output"], fourcc, args["frames"],
                        (w * 1, h * 1), True)

		zeros = np.zeros((h, w), dtype="uint8")

	#Write original frame
	if not args["original"] == "nothing":
		writer.write(frame)

	# if input queue is empty get current frame
	if inputQueue.empty():
		inputQueue.put(frame)

	# if output queue is not empty, grab detections
	if not outputQueue.empty():
		detections = outputQueue.get()

	# check if there are detections
	if detections is not None and args["detection"] == "yes":
		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# get the confidence level for the detection
			confidence = detections[0, 0, i, 2]

			# ensure minimum confidence
			if confidence < args["confidence"]:
				continue

			# if confident extract x and y coordinates detection in a box shape
			idx = int(detections[0, 0, i, 1])
			dims = np.array([fW, fH, fW, fH])
			box = detections[0, 0, i, 3:7] * dims
			(startX, startY, endX, endY) = box.astype("int")

			# draw a rectangle on the detection
			if args["draw"] == "yes":
				label = "{}: {:.2f}%".format(CLASSES[idx],
					confidence * 100)
				cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 1)

			if args["fill"] == "yes":
				label = "{}: {:.2f}%".format(CLASSES[idx],
					confidence * 100)
				cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], -1)

			y = startY - 15 if startY - 15 > 15 else startY + 15

			# some math for blur padding
			#if args["blur"] == "yes":
			if startX < 1:
				sx = 1
			else:
				if startX - BlurPadding > 0:
					sx = startX - BlurPadding
				else:
					sx = 1

			if startY < 1:
				sy = 1
			else:
				if startY - BlurPadding > 0:
					sy = startY - BlurPadding
				else:
					sy = 1

			if endX < 1:
				ex = 1
			else:
				if endX + BlurPadding > 0:
					ex = endX + BlurPadding
				else:
					ex = 1

			if endY < 1:
				ey = 1
			else:
				if endY + BlurPadding > 0:
					ey = endY + BlurPadding
				else:
					ey = 1

			# draw a label on the detection
			if args["labels"] == "yes":
				cv2.putText(frame, label, (sx, (sy - 10)),
                                	cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

			# apply blur
			if args["blur"] == "yes":
				Blurring = cv2.blur(frame[sy:ey,sx:ex], (args["blurlevel"],args["blurlevel"]));
				frame[sy:ey,sx:ex] = Blurring;

			# apply median blur
			if args["blurmedian"] == "yes":
				Blurring = cv2.medianBlur(frame[sy:ey,sx:ex], args["blurlevel"]);
				frame[sy:ey,sx:ex] = Blurring;

			# apply gaussian blur
			if args["blurgaussian"] == "yes":
				Blurring = cv2.GaussianBlur(frame[sy:ey,sx:ex], (args["blurlevel"],args["blurlevel"]), 0);
				frame[sy:ey,sx:ex] = Blurring;

			if args["warping"] == "yes":
				#frame = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
				rows, cols = frame.shape
				img_output = np.zeros(frame.shape, dtype=img.dtype)
				for i in range(rows):
					for j in range(cols):
						offset_x = int(25.0 * math.sin(2 * 3.14 * i / 180))
						offset_y = 0
						if j+offset_x < rows:
							img_output[i,j] = frame[i,(j+offset_x)%cols]
						else:
							img_output[i,j] = 0

				Blurring = cv2.GaussianBlur(frame[sy:ey,sx:ex], (args["blurlevel"],args["blurlevel"]), 0);
				frame[sy:ey,sx:ex] = Blurring;

	# show the current frame
	if args["showlocal"] == "yes":
		cv2.imshow("Frame", frame)

	# write video stream to a file
	if not args["output"] == "nothing":
		writer2.write(frame)

#	while testing < 10:
#		time.sleep(5.0)
#		os.system("ffmpeg -re -i ~/test.avi http://localhost:8090/feed2.ffm")
#		testing = 50

	# define break key 'q'
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

	# timer
	timercheck = default_timer() - start
	if not int(timercheck) < args["timer"] and not args["timer"] == 9999:
		break

	# update the FPS counter
	fps.update()

# display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# write logging
if not args["logging"] == "nothing":
	with open(args["logging"], "a") as myfile:
		print("[INFO] Writing to log")
		myfile.write("Time:{:.2f}".format(fps.elapsed()) + " FPS:{:.2f}".format(fps.fps()))
		myfile.write("\n")

# cleanup
cv2.destroyAllWindows()
vs.stop()
writer.release()
writer2.release()
p2.close()
p3.close()
#stream.terminate()
