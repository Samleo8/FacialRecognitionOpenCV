import dlib # dlib for accurate face detection
import cv2 # opencv
import imutils # helper functions from pyimagesearch.com
import argparse #parse arguments in the command line of python
import numpy as np

# Booleans
faceDetect, bodyDetect = True, False
grayscale = True

# Face detector
face_detector = dlib.get_frontal_face_detector()

# Body detector
'''
body_detector = cv2.HOGDescriptor()
body_detector.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
'''
body_detector = cv2.CascadeClassifier('_data/haarcascade_fullbody.xml')

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
		idxs = np.delete(idxs, np.concatenate(([last], where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

# Image Processing
def process_image(_frame):
    global faceDetect, bodyDetect, grayscale

    gray = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)

    # Make copies of the frame for transparency processing
    if grayscale:
        output = gray
    else:
        output = _frame

    if faceDetect:
        # detect faces in the gray scale frame
        face_rects = face_detector(gray, 0)

        # loop over the face detections
        for i, d in enumerate(face_rects):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()

            cv2.rectangle(output,(x1,y1),(x1+w,y1+h),(255,0,0),2)

        text = "{} face(s) found".format(len(face_rects))
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

        #output = imutils.resize(output, original_width)

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
        if key == ord("q") or key == 27:
            #esc or q to quit
            break

    stream.release()
    if video_writer is not None:
        print("Releasing video...")
        video_writer.release()
    cv2.destroyAllWindows()

def main():
    global faceDetect, bodyDetect, grayscale, record_video

    # Argument Parsing (Command line)
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--bodyDetect", action='store_true', help="Detect full bodies as well")
    ap.add_argument("-c", "--color", action='store_false', help="Show video in colour")
    ap.add_argument("-r", "--record", action='store_true', help="Record video")
    args = vars(ap.parse_args())

    # Set global variables accordingly
    faceDetect = True
    bodyDetect = args["bodyDetect"]
    grayscale = args["color"]
    record_video = args["record"]

    # Grab video from webcam (video_path=0)
    # or video (video_path=/path/to/video)
    video_path = 0#"TownCentreVideo.avi"
    process_video(video_path)

if __name__ == '__main__':
    main()
