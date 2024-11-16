import cv2 as cv
import os
import time

listed_person = []
DIR = r"persons"
HAAR_CASCADE_PATH = "front_face/haar_face.xml"
MODEL_PATH = 'front_face/face_trained.yml'

haar_cascade = cv.CascadeClassifier(HAAR_CASCADE_PATH)
if haar_cascade.empty():
    raise IOError(f"Haar cascade XML file not found or failed to load from {HAAR_CASCADE_PATH}")

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(MODEL_PATH)

people = [name for name in os.listdir(DIR) if os.path.isdir(os.path.join(DIR, name))]

cap = cv.VideoCapture(0)

recognized_person = None
start_time = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture video.")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=12)

    frame_flipped = cv.flip(frame, 1)

    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y + h, x:x + w]

        label, confidence = face_recognizer.predict(faces_roi)

        if 0 <= label < len(people) and confidence > 10:
            if recognized_person == people[label]:
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time >= 5:
                    print(f"Person Recognized: {people[label]}")
                    listed_person.append(people[label])
            else:
                recognized_person = people[label]
                start_time = time.time()
        else:
            recognized_person = None
            start_time = None

        label_text = f"{people[label]} ({int(confidence)})" if 0 <= label < len(people) else "Unknown"
        flipped_x = frame.shape[1] - (x + w)
        flipped_y = y

        cv.rectangle(frame_flipped, (flipped_x, flipped_y), (flipped_x + w, flipped_y + h), (0, 255, 0), thickness=2)
        cv.putText(frame_flipped, label_text, (flipped_x, flipped_y - 10), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

    cv.imshow("Live Face Recognition (Mirror)", frame_flipped)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

final_list = list(set(listed_person))
cap.release()
cv.destroyAllWindows()

for persons in final_list:
    print(persons)
