import cv2 as cv
import os
import time

MODEL_PATH = "front_face/face_trained.yml"
HAAR_CASCADE_PATH = "front_face/haar_face.xml"
PERSONS_DIR = "persons"

haar_cascade = cv.CascadeClassifier(HAAR_CASCADE_PATH)
if haar_cascade.empty():
    raise IOError(f"Haar cascade XML file not found at {HAAR_CASCADE_PATH}")

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(MODEL_PATH)

people = [name for name in os.listdir(PERSONS_DIR) if os.path.isdir(os.path.join(PERSONS_DIR, name))]

all_captured_names = set()
recognized_names_temp = []
recognized_person = None
start_time = None
frame_border_color = (0, 0, 255) 

cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture video.")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)

    recognized_names_temp = []
    for (x, y, w, h) in faces_rect[:3]:
        face_roi = gray[y:y + h, x:x + w]
        label, confidence = face_recognizer.predict(face_roi)

        if 0 <= label < len(people) and confidence < 100:
            name = people[label]
            if recognized_person == name:
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time >= 2:
                    if name not in recognized_names_temp:
                        recognized_names_temp.append(name)
                        all_captured_names.add(name)
                        frame_border_color = (0, 255, 0) 
                        print(f"the present person is {name}")
            else:
                recognized_person = name
                start_time = time.time()
        else:
            recognized_person = None
            start_time = None



    if not recognized_names_temp:
        frame_border_color = (0, 0, 255)

    recognized_text = f"Recognized: {', '.join(recognized_names_temp) if recognized_names_temp else 'None'}"
    cv.putText(frame, recognized_text, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, frame_border_color, 2)

    frame = cv.copyMakeBorder(frame, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=frame_border_color)

    cv.imshow("Live Face Recognition (With Border and Names)", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

print("Captured names:", list(all_captured_names))
