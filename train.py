'''
SETUP

In the 'dataset/' folder, create a folder with the name of the person.
Images of the person's face used for training should be inside that folder.

USAGE

python3 train.py [-m|--model hog(default)|cnn] [-t|--trainAgain]

NOTES

CNN (convolutional neural network) is slower but has a better accuracy, whereas HOG (Histogram of Oriented Gradients) is used as the default but has poor accuracy.

'''

# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="hog",
	help="Face detection model to use: either `cnn` or `hog` (default: faster, less accurate)")
ap.add_argument("-t", "--trainAgain", type=str, default="",
	help="What folders would you like to train again (space separated)?")
args = vars(ap.parse_args())

# grab the paths to the input images in our dataset
dataset_path = "dataset"
print("[INFO] Getting faces from `"+dataset_path+"` folder...")
imagePaths = list(paths.list_images(dataset_path))

# initialize the list of known encodings and known names
# but don't redo what's already known from the dataset
# unless we explicitly tell the program to train again
encodingFile = "encodings.pickle"

knownEncodings = []
knownNames = []

currentData = pickle.loads(open(encodingFile, "rb").read())
currentEncodings = currentData["encodings"]
currentNames = currentData["names"]

trainAgain = args["trainAgain"].split(" ")
toRetrain, dontTrain = {}, {}
for n in trainAgain: toRetrain[n]=True

for (i, name) in enumerate(currentNames):
	if name in toRetrain: continue
	knownEncodings.append(currentEncodings[i])
	knownNames.append(currentNames[i])

uniqueNames = list(set(knownNames))
for n in uniqueNames: dontTrain[n]=True

print("[INFO] Names which we will NOT re-train: "+str(uniqueNames).replace("\'","").replace("[","").replace("]",""))
print("[INFO] If you would like to retrain them, use python3 train.py -t [names to train, space-separated]")

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	name = imagePath.split(os.path.sep)[-2]

	if name in dontTrain: continue

	print("[INFO] Processing image {}/{} for {}".format(i + 1,len(imagePaths),name))


	# load the input image and convert it from BGR (OpenCV ordering)
	# to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(rgb,model=args["model"])

	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)

	# loop over the encodings
	for encoding in encodings:
	# add each encoding + name to our set of known names and encodings
		knownEncodings.append(encoding)
		knownNames.append(name)

# dump the facial encodings + names to disk
print("[INFO] Serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(encodingFile, "wb")
f.write(pickle.dumps(data))
f.close()
print("[INFO] Complete!")
