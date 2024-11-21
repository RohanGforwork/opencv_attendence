from flask import Flask, render_template, Response, jsonify
import cv2 as cv
import os
from flask_cors import CORS
import threading
import time

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

# Separate thread to handle video capture
def capture_video():
    global recognized_names, detected_faces
    cap = cv.VideoCapture(0)

    recognized_person = None
    start_time = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture video.")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)

        detected_faces = len(faces_rect) > 0
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
                else:
                    recognized_person = name
                    start_time = time.time()
            else:
                recognized_person = None
                start_time = None
                if "Unknown" not in recognized_names_temp:
                    recognized_names_temp.append("Unknown")
                    all_captured_names.add("Unknown")

        recognized_names = recognized_names_temp

        # Return processed frame for video feed
        ret, buffer = cv.imencode('.jpg', frame)
        if ret:
            frame = buffer.tobytes()
            video_feed_buffer = (b'--frame\r\n'
                                 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            # Store or send this buffer to the front end

    cap.release()

# This function starts the video capture in a separate thread
def start_video_capture():
    video_thread = threading.Thread(target=capture_video)
    video_thread.daemon = True
    video_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        cap = cv.VideoCapture(0)
        recognized_names_temp = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale for face detection
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)

            # Detect faces and perform recognition
            recognized_names_temp.clear()  # Clear previous names
            for (x, y, w, h) in faces_rect[:3]:
                face_roi = gray[y:y + h, x:x + w]
                label, confidence = face_recognizer.predict(face_roi)

                if 0 <= label < len(people) and confidence < 100:
                    name = people[label]
                    recognized_names_temp.append(name)
                else:
                    recognized_names_temp.append("Unknown")

            # Draw rectangles and names
            for (x, y, w, h), name in zip(faces_rect[:3], recognized_names_temp):
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(frame, name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Convert frame to JPEG for streaming
            ret, buffer = cv.imencode('.jpg', frame)
            if not ret:
                continue

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/get_recognized_names', methods=['GET'])
def get_recognized_names():
    global recognized_names, detected_faces
    return jsonify({
        "names": ", ".join(recognized_names),
        "detected": detected_faces
    })

if __name__ == '__main__':
    start_video_capture()  # Start video capture in a separate thread
    app.run(debug=False)
