from flask import Flask, render_template, Response, jsonify
import cv2 as cv
import os
from flask_cors import CORS
from datetime import datetime
import atexit
import time
import numpy as np
import threading
import dlib
from imutils import face_utils

app = Flask(__name__)
CORS(app)

MODEL_PATH = "front_face/face_trained.yml"
HAAR_CASCADE_PATH = "front_face/haar_face.xml"
PERSONS_DIR = "persons"

haar_cascade = cv.CascadeClassifier(HAAR_CASCADE_PATH)
if haar_cascade.empty():
    raise IOError(f"Haar cascade XML file not found at {HAAR_CASCADE_PATH}")

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(MODEL_PATH)

people = [name for name in os.listdir(PERSONS_DIR) if os.path.isdir(os.path.join(PERSONS_DIR, name))]

recognized_names = []
detected_faces = False
all_captured_names = set()

class VideoStream:
    def __init__(self):
        self.cap = cv.VideoCapture(0)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(target=self.update_frames, daemon=True)
        self.thread.start()

    def update_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False
        self.cap.release()

video_stream = VideoStream()





predictor_path = "shape_predictor_68_face_landmarks.dat"  # Download this model
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)

# Constants for blink detection
EYE_AR_THRESH = 0.2  # Eye Aspect Ratio threshold
EYE_AR_CONSEC_FRAMES = 3  # Minimum consecutive frames for a blink

# Indices for the eyes in the 68-point facial landmarks
(left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

blink_counter = 0
blink_confirmed = False

def calculate_eye_aspect_ratio(eye):
    # Compute the distances between the vertical eye landmarks
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # Compute the distance between the horizontal eye landmarks
    C = np.linalg.norm(eye[0] - eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear



def generate_frames():
    global recognized_names, detected_faces, all_captured_names, blink_counter, blink_confirmed

    recognized_person = None
    start_time = None
    frame_border_color = (0, 0, 255)

    while True:
        frame = video_stream.get_frame()
        if frame is None:
            continue

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        detected_faces = len(faces) > 0
        recognized_names_temp = []

        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            face_roi = gray[y:y + h, x:x + w]

            # Predict face label
            label, confidence = face_recognizer.predict(face_roi)

            if 0 <= label < len(people) and confidence < 100:
                name = people[label]

                # Blink detection
                shape = shape_predictor(gray, face)
                shape = face_utils.shape_to_np(shape)

                left_eye = shape[left_start:left_end]
                right_eye = shape[right_start:right_end]

                left_ear = calculate_eye_aspect_ratio(left_eye)
                right_ear = calculate_eye_aspect_ratio(right_eye)

                ear = (left_ear + right_ear) / 2.0

                if ear < EYE_AR_THRESH:
                    blink_counter += 1
                else:
                    if blink_counter >= EYE_AR_CONSEC_FRAMES:
                        blink_confirmed = True
                        blink_counter = 0

                if blink_confirmed:
                    if recognized_person == name:
                        if start_time is None:
                            start_time = time.time()
                        elif time.time() - start_time >= 2:
                            if name not in recognized_names_temp:
                                recognized_names_temp.append(name)
                                all_captured_names.add(name)
                                frame_border_color = (0, 255, 0)
                    else:
                        recognized_person = name
                        start_time = time.time()
                else:
                    recognized_person = None
                    start_time = None

                    if "Unknown" not in recognized_names_temp:
                        recognized_names_temp.append("Unknown")
                        all_captured_names.add("Unknown")
            else:
                recognized_person = None
                start_time = None

                if "Unknown" not in recognized_names_temp:
                    recognized_names_temp.append("Unknown")
                    all_captured_names.add("Unknown")

        recognized_names = recognized_names_temp

        if not recognized_names_temp:
            frame_border_color = (0, 0, 255)

        # Draw border
        frame = cv.copyMakeBorder(frame, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=frame_border_color)
        ret, buffer = cv.imencode('.jpg', frame)

        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_recognized_names', methods=['GET'])
def get_recognized_names():
    global recognized_names, detected_faces
    return jsonify({
        "names": ", ".join(recognized_names),
        "detected": detected_faces
    })

def on_exit():
    global all_captured_names
    print("\n--- Session Summary ---")
    print("All captured names:")
    print(", ".join(all_captured_names))

atexit.register(on_exit)

if __name__ == '__main__':
    app.run(debug=False,threaded = True,host='0.0.0.0', port=5000)