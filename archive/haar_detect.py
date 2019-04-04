import cv2
import numpy as np

# CASCADES
uppBodyCasPath = "_haar/haarcascade_upperbody.xml"
lowBodyCasPath = "_haar/haarcascade_lowerbody.xml"
faceCasPath = [
    "_haar/haarcascade_frontalface_default.xml",
    "_haar/haarcascade_frontalface_alt.xml",
    "_haar/haarcascade_frontalface_alt2.xml",
    "_haar/haarcascade_frontalface_alt_tree.xml"
]

uppBody_cascade = cv2.CascadeClassifier(uppBodyCasPath)
lowBody_cascade = cv2.CascadeClassifier(lowBodyCasPath)
face_cascade = cv2.CascadeClassifier(faceCasPath[0])

# FUNCTION: process_image(frame)
#   Processes image to detect uppper and lower body, and faces
#   returns frame
def process_image(_frame, grayscale=True):
    img_gray = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)

    #Actual Detection
    #(image, rejectLevels, levelWeights)
    uppBody = uppBody_cascade.detectMultiScale(img_gray, 1.2, 5)
    lowBody = lowBody_cascade.detectMultiScale(img_gray, 1.2, 5)
    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)

    if grayscale:
        frame = img_gray
    else:
        frame = _frame

    #BODY RECOGNITION
    for (x,y,w,h) in uppBody:
    	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    for (x,y,w,h) in lowBody:
    	cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),2)

    #FACE RECOGNITION
    for (x,y,w,h) in faces:
    	cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    text = "{} face(s) found".format(len(faces))
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    text = "{} upper bodies found".format(len(uppBody))
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    _height, _width = frame.shape[:2]
    text = "Q/Esc to Quit"
    cv2.putText(frame, text, (_width-130, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    #IS THERE A ___?
    '''
    n = len(faces)
    if n:
        print(str(n)+" faces detected!")

    n = len(uppBody)
    if n:
        print(str(n)+" (upper) bodies detected!")
    '''

    return frame;

def show_webcam(mirror=False,grayscale=True):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)

        # Our operations on the frame come here
        img_proc = process_image(img,grayscale)

        cv2.imshow('Processed Image', img_proc)
        key = cv2.waitKey(1)
        if key == 27 or key == ord("q"):
            break  # esc or 'q' to quit
    cv2.destroyAllWindows()

def main():
    show_webcam(mirror=True,grayscale=True)

    '''
    _img = process_image(cv2.imread("test.jpeg"))
    cv2.imwrite('processed_photo.jpg',_img)
    '''

if __name__ == '__main__':
    main()
