# #pylint:disable=no-member

# import numpy as np
# import cv2 as cv
# DIR = r"simpsons_dataset"

# haar_cascade = cv.CascadeClassifier("opencv-course/Section #3 - Faces/haar_face.xml")
# people = []
# for characters in DIR:
#     people.append(characters)
# # features = np.load('features.npy', allow_pickle=True)
# # labels = np.load('labels.npy')

# face_recognizer = cv.face.LBPHFaceRecognizer_create()
# face_recognizer.read('face_trained.yml')

# img = cv.imread(r'kaggle_simpson_testset\kaggle_simpson_testset\charles_montgomery_burns_21.jpg')

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Person', gray)

# # Detect the face in the image
# faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

# for (x,y,w,h) in faces_rect:
#     faces_roi = gray[y:y+h,x:x+w]

#     label, confidence = face_recognizer.predict(faces_roi)
#     print(f'Label = {people[label]} with a confidence of {confidence}')

#     cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
#     cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

# cv.imshow('Detected Face', img)

# cv.waitKey(0)
# pylint:disable=no-member
import cv2
print(cv2.__version__)

import numpy as np
import cv2 as cv
import os

DIR = r"persons"
haar_cascade = cv.CascadeClassifier("front_face/haar_face.xml")

people = [name for name in os.listdir(DIR) if os.path.isdir(os.path.join(DIR, name))]

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('front_face/face_trained.yml')
img = cv.imread(r'persons\gagan\frame_1.jpg')

if img is None:
    print("Error: Test image not found!")
else:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Person', gray)

    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)
    if len(faces_rect) == 0:
        print("No faces detected.")
    else:
        for (x, y, w, h) in faces_rect:
            faces_roi = gray[y:y+h, x:x+w]
            label, confidence = face_recognizer.predict(faces_roi)

            if 0 <= label < len(people):
                print(f'Label = {people[label]} with a confidence of {confidence}')
                cv.putText(img, str(people[label]), (x, y-10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
                cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
            else:
                print("Unknown label:", label)
     
    print(f"the confidence score is {confidence}")
    cv.imshow('Detected Face', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
