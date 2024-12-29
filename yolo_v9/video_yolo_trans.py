import cv2
from ultralytics import YOLO
import time
import numpy as np

MODEL_PATH = r"C:\Users\Roshan\runs\classify\train16\weights\last.pt"

model = YOLO(MODEL_PATH)

recognized_person = None
start_time = None
all_captured_names = set()
recognized_names_temp = set()
frame_border_color = (0, 0, 255)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    if results and results[0].probs is not None:
        name_dict = results[0].names
        probs = results[0].probs.data.cpu().numpy()
        top_class_index = np.argmax(probs)
        top_class = name_dict[top_class_index]
        top_confidence = probs[top_class_index]

        if top_confidence > 0.8:
            if recognized_person == top_class:
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time >= 3:
                    if top_class not in recognized_names_temp:
                        recognized_names_temp.add(top_class)
                        all_captured_names.add(top_class)
                        frame_border_color = (0, 255, 0)
            else:
                recognized_person = top_class
                start_time = time.time()
        else:
            recognized_person = None
            start_time = None
    else:
        recognized_person = None
        start_time = None

    if recognized_person is None:
        frame_border_color = (0, 0, 255)

    recognized_text = f"Recognized: {recognized_person if recognized_person else 'None'}"
    cv2.putText(frame, recognized_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, frame_border_color, 2)
    frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=frame_border_color)

    cv2.imshow("YOLO Real-Time Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(list(all_captured_names))
