import os
import cv2 as cv
import numpy as np

DIR = 'persons'
HAAR_CASCADE_PATH = 'front_face/haar_face.xml'
BATCH_SIZE = 32
EPOCHS = 10

people = [person for person in os.listdir(DIR) if os.path.isdir(os.path.join(DIR, person))]
haar_cascade = cv.CascadeClassifier(HAAR_CASCADE_PATH)

if haar_cascade.empty():
    raise IOError(f"Haar cascade XML file not found or failed to load from {HAAR_CASCADE_PATH}")

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            if img_array is None:
                continue

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)

face_recognizer.save('front_face/face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)


print("Training completed and models saved successfully.")