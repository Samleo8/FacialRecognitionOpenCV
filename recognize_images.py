import dlib # dlib for accurate face detection
import cv2 # opencv
import imutils # helper functions from pyimagesearch.com
from imutils import paths
import argparse # parse arguments in the command line of python
import numpy as np # numpy for various functions
import face_recognition # for accurate face recognition
import pickle # to parse the file that contains the learnt models

# Facial recognition
modelData = None
encodingFile = "encodings.pickle"
threshold = 30.0

def recognise_faces(_frame):
	global encodingFile, modelData

	output = _frame.copy()

	names = []
	rgb = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)

	boxes = face_recognition.face_locations(rgb,model="hog")
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

		print(counts)

		# update the list of names
		if percent <= threshold:
			names.append({"name":"unknown","percent":0})
		else:
			names.append({"name":name,"percent":percent})

	#print(names)

	# loop over the recognized faces
	for ((top, right, bottom, left), name_obj) in zip(boxes, names):
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

	return output

if __name__ == '__main__':
	modelData = pickle.loads(open(encodingFile, "rb").read())

	toRecoPath = "toReco"

	imagePaths = list(paths.list_images(toRecoPath))
	#print(imagePaths)

	for img in imagePaths:
		image = cv2.imread(img)
		#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		out = recognise_faces(image)

		#cv2.imshow("Image "+img,out)
		outputPath = "_reco/"+img.split(toRecoPath+"/")[1]
		print("[INFO] Complete! \nProcessed image can be found at: "+outputPath+"\n")
		cv2.imwrite(outputPath, out)

	#cv2.waitKey(0)
	cv2.destroyAllWindows()
