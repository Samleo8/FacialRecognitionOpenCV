# import the necessary packages
from imutils import paths
import dlib
import argparse
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-o", "--output", default='webcam_output', help="Path to save image files")
ap.add_argument("-n", "--total", default=50, help="Total number of pictures to save")
ap.add_argument("-a", "--prepend", default="", help="Prepend to filename")

args = vars(ap.parse_args())

# Set global variables accordingly
prepend = args["prepend"]
output_dir = args["output"]
max_total = min(100, int(args["total"]))

try:
	os.mkdir(output_dir)
except Exception as e:
	print("[WARNING] Output folder already exists, or cannot be made.")

face_detector = dlib.get_frontal_face_detector()

stream = cv2.VideoCapture(0)

total = 0
xpadd = 50
ypadd = 100

# Read frame by frame from video
while stream.isOpened():
	grabbed, frame = stream.read()

	# detect faces in the gray scale frame
	face_rects = face_detector(frame, 0)

	if(len(face_rects)): d = face_rects[0]
	else: continue

	# loop over the face detections
	# loop over the face detections
	x1, y1, x2, y2 = d.left(), d.top(), d.right() + 1, d.bottom() + 1

	out = frame[y1-ypadd:y2+ypadd, x1-xpadd:x2+xpadd]

	if grabbed==False or total==max_total or total>1000: break

	# press q to break out of the loop
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q") or key == ord("w") or key == 27:
		#esc or q to quit
		break

	# Show the frame
	cv2.imshow("Face Detect", out)

	# Output the frame into the output folder
	img_path = output_dir+"/"+prepend+str(total).zfill(3)+".jpg"
	cv2.imwrite(img_path, out)

	total += 1

stream.release()
cv2.destroyAllWindows()
