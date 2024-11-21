from flask import Flask, render_template, Response, jsonify
import cv2 as cv
import os
from flask_cors import CORS
from datetime import datetime
import atexit
import time
import numpy as np
app = Flask(__name__)
CORS(app)

# Paths to the trained model and Haar Cascade
MODEL_PATH = "front_face/face_trained.yml"
HAAR_CASCADE_PATH = "front_face/haar_face.xml"
PERSONS_DIR = "persons"

# Load the Haar Cascade and face recognizer
haar_cascade = cv.CascadeClassifier(HAAR_CASCADE_PATH)
if haar_cascade.empty():
    raise IOError(f"Haar cascade XML file not found at {HAAR_CASCADE_PATH}")

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(MODEL_PATH)

# Load the names of the individuals
people = [name for name in os.listdir(PERSONS_DIR) if os.path.isdir(os.path.join(PERSONS_DIR, name))]

# Global variables
recognized_names = []  # List to store recognized names for the current frame
detected_faces = False  # Flag for whether faces are detected
all_captured_names = set()  # Set to store all unique names captured during the session


# def generate_frames():
#     global recognized_names, detected_faces, all_captured_names
#     cap = cv.VideoCapture(0)  # Access webcam (change index if needed)
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Unable to capture video.")
#             break

#         gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#         faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)

#         detected_faces = len(faces_rect) > 0  # Set flag if faces are detected
#         recognized_names_temp = []  # Temporary list for the current frame

#         for (x, y, w, h) in faces_rect[:3]:  # Process at most 3 faces
#             face_roi = gray[y:y+h, x:x+w]
#             label, confidence = face_recognizer.predict(face_roi)
#             if 0 <= label < len(people) and confidence > 10:
#                 name = people[label]
#                 if name not in recognized_names_temp:
#                     recognized_names_temp.append(name)
#                     all_captured_names.add(name)  # Add to the global set
#             else:
#                 if "Unknown" not in recognized_names_temp:
#                     recognized_names_temp.append("Unknown")
#                     all_captured_names.add("Unknown")  # Add "Unknown" to the global set

#         recognized_names = recognized_names_temp  # Update the global recognized names list

#         # Draw rectangles and labels around detected faces
#         for (x, y, w, h), name in zip(faces_rect[:3], recognized_names_temp):
#             cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
#             cv.putText(frame, name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#         # Encode the frame as JPEG for streaming
#         ret, buffer = cv.imencode('.jpg', frame)
#         if not ret:
#             continue
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

#     cap.release()
# def generate_frames():
#     global recognized_names, detected_faces, all_captured_names
#     cap = cv.VideoCapture(0)  # Access webcam (change index if needed)


#     # cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
#     # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)


#     # cap.set(cv.CAP_PROP_FPS, 30)  # Adjust to a lower frame rate (e.g., 15 FPS)

#     recognized_person = None  # To track the current recognized person
#     start_time = None  # To track when the detection of a person started

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Unable to capture video.")
#             break

#         gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#         faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)

#         detected_faces = len(faces_rect) > 0  # Set flag if faces are detected
#         recognized_names_temp = []  # Temporary list for the current frame

#         for (x, y, w, h) in faces_rect[:1]:  # Process at most 3 faces
#             face_roi = gray[y:y + h, x:x + w]
#             label, confidence = face_recognizer.predict(face_roi)

#             if 0 <= label < len(people) and confidence < 100:  # Adjusted confidence threshold
#                 name = people[label]

#                 # Check if the detected person is the same as the previously recognized person
#                 if recognized_person == name:
#                     if start_time is None:
#                         start_time = time.time()  # Start timing
#                     elif time.time() - start_time >= 2:  # Check if detection lasts 2+ seconds
#                         if name not in recognized_names_temp:
#                             print(f"Person Recognized: {name}")
#                             recognized_names_temp.append(name)
#                             all_captured_names.add(name)  # Add to the global set
#                 else:
#                     recognized_person = name
#                     start_time = time.time()  # Reset timing
#             else:
#                 recognized_person = None
#                 start_time = None

#                 if "Unknown" not in recognized_names_temp:
#                     recognized_names_temp.append("Unknown")
#                     all_captured_names.add("Unknown")  # Add "Unknown" to the global set

#         recognized_names = recognized_names_temp  # Update the global recognized names list

#         # Draw rectangles and labels around detected faces
#         for (x, y, w, h), name in zip(faces_rect[:3], recognized_names_temp):
#             cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
#             cv.putText(frame, name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#         # Encode the frame as JPEG for streaming
#         ret, buffer = cv.imencode('.jpg', frame)
#         if not ret:
#             continue
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

#     cap.release()

yolo_net = cv.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")  # Replace with your YOLO weights and config
yolo_net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
yolo_net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)  # Use GPU if available: DNN_TARGET_CUDA

# Load YOLO classes (if applicable)
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Modify the generate_frames function
def generate_frames():
    global recognized_names, detected_faces, all_captured_names
    cap = cv.VideoCapture(0)  # Access webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture video.")
            break

        height, width, _ = frame.shape

        # Preprocess the frame for YOLO
        blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        yolo_net.setInput(blob)
        detections = yolo_net.forward(output_layers)

        recognized_names_temp = []  # Temporary list for current frame
        detected_faces = False

        # Process YOLO detections
        for detection in detections:
            for object_detection in detection:
                scores = object_detection[5:]  # Confidence scores for each class
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:  # Confidence threshold
                    detected_faces = True

                    # Get bounding box coordinates
                    center_x = int(object_detection[0] * width)
                    center_y = int(object_detection[1] * height)
                    w = int(object_detection[2] * width)
                    h = int(object_detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Add your face recognition logic here (if applicable)
                    name = "Person"  # Placeholder for recognized name
                    if name not in recognized_names_temp:
                        recognized_names_temp.append(name)
                        all_captured_names.add(name)

                    # Draw bounding box and label
                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv.putText(frame, name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        recognized_names = recognized_names_temp  # Update the global recognized names list

        # Encode the frame as JPEG for streaming
        ret, buffer = cv.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()



@app.route('/')
def index():
    return render_template('index.html')  # Ensure index.html exists in templates folder


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_recognized_names', methods=['GET'])
def get_recognized_names():
    global recognized_names, detected_faces
    return jsonify({
        "names": ", ".join(recognized_names),  # Join names with commas
        "detected": detected_faces
    })


# Function to execute when the application quits
def on_exit():
    global all_captured_names
    print("\n--- Session Summary ---")
    print("All captured names:")
    print(", ".join(all_captured_names))


# Register the exit handler
atexit.register(on_exit)

if __name__ == '__main__':
    app.run(debug=False)