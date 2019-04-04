'''
USAGE:

python3 recognize.py [ -b | --bodyDetect ] [ -c | --color ] [ -s | --save ] [ -r , --recogniseFaces ] [ -a | --accurateRecognise ]

DEPENDENCIES:

dlib, opencv, imutils, argparse, face_recognition

REFERENCES:

Code adapted from https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/
'''

import dlib # dlib for accurate face detection
import cv2 # opencv
import imutils # helper functions from pyimagesearch.com
import argparse # parse arguments in the command line of python
import numpy as np # numpy for various functions
import face_recognition # for accurate face recognition
import pickle # to parse the file that contains the learnt models

# Global Booleans
faceDetect, bodyDetect = True, False
fastFaceRecognise, faceRecognise = False, False
grayscale = True

# Face detector
face_detector = dlib.get_frontal_face_detector()

# Body detector
'''
body_detector = cv2.HOGDescriptor()
body_detector.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
'''
body_detector = cv2.CascadeClassifier('haarcascade_upperbody.xml')

# Facial recognition
modelData = None
encodingFile = "encodings.pickle"

# Video Saving
video_writer = None
record_video = False

# Non-Maximum Suppression (Fast)
# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

# Facial recognition
modelData = None
encodingFile = "encodings.pickle"
threshold = 20.00

def recognise_faces(_frame):
	global encodingFile, modelData, fastFaceRecognise

	if _frame is None: return (_frame, 0)

	if fastFaceRecognise:
		_copiedFrame = cv2.resize(_frame, (0, 0), fx=0.25, fy=0.25)
	else:
		_copiedFrame = _frame.copy()

	output = _frame

	names = []
	rgb = cv2.cvtColor(_copiedFrame, cv2.COLOR_BGR2RGB) # convert to RGB

	boxes = face_recognition.face_locations(rgb, model="hog")
	encodings = face_recognition.face_encodings(rgb, boxes)

	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(modelData["encodings"],encoding)
		name = "unknown"
		percent = 0

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = modelData["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number of
			# votes (note: in the event of an unlikely tie Python will
			# select first entry in the dictionary)
			name = max(counts, key=counts.get)
			percent = float("{0:.2f}".format(100*counts[name]/len(matchedIdxs)))

		# update the list of names
		if percent <= threshold:
			names.append({"name":"unknown","percent":0})
		else:
			names.append({"name":name,"percent":percent})

	#print(names)

	# loop over the recognized faces
	for ((top, right, bottom, left), name_obj) in zip(boxes, names):
		if fastFaceRecognise:
			# Scale back up face locations since the frame we detected in was scaled to 1/4 size
			top *= 4
			right *= 4
			bottom *= 4
			left *= 4

		# draw the predicted face name on the image
		cv2.rectangle(output, (left, top), (right, bottom), (255, 0, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15

		nm = name_obj["name"]
		percent = name_obj["percent"]

		if nm=="unknown" or percent<=threshold:
			_str = "unknown"
		else:
			_str = nm+", "+str(percent)+"%"

		cv2.putText(output, _str, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

	return (output, len(names))

# Image Processing
def process_image(_frame):
	global faceDetect, bodyDetect, grayscale

	gray = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)

	# Make copies of the frame for transparency processing
	if grayscale:
		output = gray
	else:
		output = _frame

	nFaces = 0
	if faceDetect:
		# facial recognition here
		if faceRecognise:
			output, nFaces = recognise_faces(output)
			fHeight, fWidth = output.shape[:2]

			text = "[ FACE RECOGNITION ACTIVE ]"
			_x = int(fWidth/2-len(text)*6)
			cv2.putText(output, text, (_x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
		else:
			# detect faces in the gray scale frame
			face_rects = face_detector(gray, 0)

			# loop over the face detections
			for i, d in enumerate(face_rects):
				x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()

				cv2.rectangle(output,(x1,y1),(x1+w,y1+h),(255,0,0),2)

			nFaces = len(face_rects)

		# outputs
		text = "{} face(s) found".format(nFaces)
		cv2.putText(output, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

	if bodyDetect:
		# Here we resize the image to
		# (1) reduce detection time
		# (2) improve detection accuracy
		#original_width = output.shape[1]
		#output = imutils.resize(output, width=min(400, original_width))

		#(image, rejectLevels, levelWeights)
		# levelWeights: higher implies less false positives
		body_rects = body_detector.detectMultiScale(output, 1.2, 1)

		body_rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in body_rects])
		pick = non_max_suppression_fast(body_rects,overlapThresh=0.65)

		for (xA, yA, xB, yB) in pick:
			cv2.rectangle(output, (xA, yA), (xB, yB), (0, 0, 255), 2)

		text = "{} full bodies found".format(len(pick))
		if faceDetect: _y = 40
		else: _y = 20

		cv2.putText(output, text, (10, _y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

	return output

# Video Processing
def process_video(_video_path):
	global record_video, video_writer

	#if _video_path != 0: record_video = True

	stream = cv2.VideoCapture(_video_path)

	# Read frame by frame from video
	while stream.isOpened():
		grabbed, frame = stream.read()

		# Initialise video writer if not initialised before and recording is required
		if video_writer is None and record_video:
			print("Initialised Video Recorder")
			fshape = frame.shape; fheight = fshape[0]; fwidth = fshape[1]
			fourcc = cv2.VideoWriter_fourcc(*'MJPG')
			video_writer = cv2.VideoWriter('output.avi', fourcc, 20.0, (fwidth,fheight), (not grayscale))

		out = process_image(frame)

		# Add helper text
		_height, _width = out.shape[:2]
		text = "Q/Esc to Quit"
		cv2.putText(out, text, (_width-130, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

		if grabbed==False: break

		# Show the frame
		cv2.imshow("Face Detection", out)

		# Write to output.avi
		if record_video:
			video_writer.write(out)

		# press q to break out of the loop
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q") or key == ord("w") or key == 27:
			#esc or q to quit
			break

	stream.release()
	if video_writer is not None:
		print("Releasing video...")
		video_writer.release()
	cv2.destroyAllWindows()

def main():
	global faceDetect, bodyDetect, grayscale, record_video, modelData, encodingFile, faceRecognise, fastFaceRecognise

	# Argument Parsing (Command line)
	ap = argparse.ArgumentParser()

	ap.add_argument("-b", "--bodyDetect", action='store_true', help="Detect full bodies as well")
	ap.add_argument("-c", "--color", action='store_false', help="Show video in colour")
	ap.add_argument("-s", "--save", action='store_true', help="Save video")
	ap.add_argument("-r", "--recogniseFaces", action='store_true', help="Save video")
	ap.add_argument("-a", "--accurateRecognise", action='store_false', help="Save video")

	args = vars(ap.parse_args())

	# Set global variables accordingly
	faceDetect = True
	bodyDetect = args["bodyDetect"]
	grayscale = args["color"]
	record_video = args["save"]
	faceRecognise = args["recogniseFaces"]
	fastFaceRecognise = args["accurateRecognise"]

	modelData = pickle.loads(open(encodingFile, "rb").read())

	# Grab video from webcam (video_path=0)
	# or video (video_path=/path/to/video)
	video_path = 0#"TownCentreVideo.avi"
	process_video(video_path)

if __name__ == '__main__':
	main()
