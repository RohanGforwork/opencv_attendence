import cv2
from ultralytics import YOLO
import time
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib

# Load YOLO model
MODEL_PATH = r"C:\Users\Roshan\runs\classify\train16\weights\last.pt"
model = YOLO(MODEL_PATH)

# Initialize dlib's face detector and facial landmark predictor for blinking detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye aspect ratio threshold and frames for blink detection
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 3

# Function to calculate the eye aspect ratio
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize variables for YOLO recognition
recognized_person = None
start_time = None
frame_border_color = (0, 0, 255)
CONFIDENCE_THRESHOLD = 0.93

# Dictionary to store each person's blink count and recognition time
person_blink_count = {}
person_recognition_time = {}
present_list = []

# Variables to track face movement (to avoid false blinks from static photos)
last_face_position = None
movement_threshold = 20  # Minimum pixel movement threshold to count as "movement"

# Start video capture
cap = cv2.VideoCapture(0)

# Threshold for EAR difference to count as a blink
EAR_CHANGE_THRESHOLD = 0.03  # A small change in EAR to trigger a blink
previous_ear = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = model(frame)

    # Convert frame to grayscale for dlib's face detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    # Variables to track if there is face movement
    current_face_position = None

    # Check if multiple faces are detected
    if len(rects) > 1:
        # Multiple faces detected - don't register any faces and show message
        cv2.putText(frame, "Multiple people detected, please look at the camera one at a time.", 
                    (20, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        frame_border_color = (0, 0, 255)  # Red border for multiple faces
        recognized_person = "Unknown"  # Reset recognized person if multiple faces
    else:
        # Process only if one face is detected
        if results and results[0].probs is not None:
            name_dict = results[0].names
            probs = results[0].probs.data.cpu().numpy()
            top_class_index = np.argmax(probs)
            top_class = name_dict[top_class_index]
            top_confidence = probs[top_class_index]

            if top_confidence >= CONFIDENCE_THRESHOLD:
                if recognized_person == top_class:
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= 3:  # 3 seconds of continuous recognition
                        if top_class not in person_recognition_time:
                            person_recognition_time[top_class] = time.time()
                else:
                    recognized_person = top_class
                    start_time = time.time()
                    if top_class not in person_blink_count:
                        person_blink_count[top_class] = 0
                        person_recognition_time[top_class] = 0
            else:
                recognized_person = "Unknown"
                start_time = None
                frame_border_color = (0, 0, 255)

        # Blink detection using dlib
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Get current face position (x, y)
            current_face_position = (rect.left(), rect.top(), rect.right(), rect.bottom())

            # Check if there is face movement by comparing previous and current position
            if last_face_position:
                x_diff = abs(current_face_position[0] - last_face_position[0])
                y_diff = abs(current_face_position[1] - last_face_position[1])

                if x_diff > movement_threshold or y_diff > movement_threshold:
                    # There is significant movement
                    face_moving = True
                else:
                    # No significant movement
                    face_moving = False
            else:
                # First frame, no previous position to compare
                face_moving = True

            last_face_position = current_face_position  # Update last face position

            # If face is moving, proceed with blink detection
            if face_moving:
                # Extract eye coordinates (using correct indices for the left and right eyes)
                leftEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][1]]
                rightEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][1]]

                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                # Average the eye aspect ratio
                ear = (leftEAR + rightEAR) / 2.0

                # If person is recognized and blink count is needed, track blinks for that person
                if recognized_person != "Unknown":
                    if previous_ear is not None:
                        ear_diff = abs(previous_ear - ear)
                        # If there's a significant drop in EAR, count it as a blink
                        if ear_diff > EAR_CHANGE_THRESHOLD and ear < EYE_AR_THRESH:
                            person_blink_count[recognized_person] += 1
                            print(f"Blink detected for {recognized_person}")

                    previous_ear = ear  # Update the previous EAR value

                    # Update the condition to require at least 3 blinks
                    if person_blink_count[recognized_person] >= 3:
                        if recognized_person not in present_list and time.time() - person_recognition_time.get(recognized_person, 0) >= 3:
                            present_list.append(recognized_person)
                            frame_border_color = (0, 255, 0)  # Green if present

    # Display the recognized person and blink count
    recognized_text = f"Recognized: {recognized_person}, Blinks: {person_blink_count.get(recognized_person, 0)}"
    cv2.putText(frame, recognized_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, frame_border_color, 2)

    # Draw border around the frame
    frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=frame_border_color)

    # Show the frame
    cv2.imshow("YOLO Real-Time Recognition with Liveness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print all recognized present persons
print("Present People:", present_list)
