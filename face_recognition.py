'''
This program is a real time working face recongition system using a reference image 
and see if user is matching or not matching the reference image.
'''
import threading
import os

import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

COUNTER = 0

FACE_MATCH = False

file_name = os.path.join(os.path.dirname(__file__), 'reference.jpg')
assert os.path.exists(file_name)

reference_img = cv2.imread(file_name)

def check_face(frame):
    '''Function is checking faces through webcam with reference image.'''
    global FACE_MATCH
    try:
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            FACE_MATCH = True
        else:
            FACE_MATCH = False
    except ValueError:
        FACE_MATCH = False


while True:
    ret, frame = cap.read()

    if ret:
        if COUNTER % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        COUNTER += 1

        if FACE_MATCH:
            cv2.putText(frame, "MATCH!", (20,450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20,450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
