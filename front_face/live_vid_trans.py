# import cv2 as cv
# import os

# DIR = r"persons"  
# HAAR_CASCADE_PATH = "opencv-course/Section #3 - Faces/haar_face.xml"
# MODEL_PATH = 'opencv-course/Section #3 - Faces/face_trained.yml'  

# haar_cascade = cv.CascadeClassifier(HAAR_CASCADE_PATH)
# if haar_cascade.empty():
#     raise IOError(f"Haar cascade XML file not found or failed to load from {HAAR_CASCADE_PATH}")

# face_recognizer = cv.face.LBPHFaceRecognizer_create()
# face_recognizer.read(MODEL_PATH)

# people = [name for name in os.listdir(DIR) if os.path.isdir(os.path.join(DIR, name))]

# cap = cv.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read() 
#     if not ret:
#         print("Error: Unable to capture video.")
#         break
    
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
#     faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)
    
#     for (x, y, w, h) in faces_rect:
#         faces_roi = gray[y:y+h, x:x+w]

#         label, confidence = face_recognizer.predict(faces_roi)
        
#         if 0 <= label < len(people):
#             label_text = f"{people[label]} ({int(confidence)})"
#         else:
#             label_text = "Unknown"
#         cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
#         cv.putText(frame, label_text, (x, y-10), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
#     frame = cv.flip(frame,1)
#     cv.imshow("Live Face Recognition", frame)
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv.destroyAllWindows()
import cv2 as cv
import os

DIR = r"persons"  
HAAR_CASCADE_PATH = "opencv-course/Section #3 - Faces/haar_face.xml"
MODEL_PATH = 'opencv-course/Section #3 - Faces/face_trained.yml'  

haar_cascade = cv.CascadeClassifier(HAAR_CASCADE_PATH)
if haar_cascade.empty():
    raise IOError(f"Haar cascade XML file not found or failed to load from {HAAR_CASCADE_PATH}")

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(MODEL_PATH)

people = [name for name in os.listdir(DIR) if os.path.isdir(os.path.join(DIR, name))]

cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read() 
    if not ret:
        print("Error: Unable to capture video.")
        break
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)
    
    # Flip the frame horizontally
    frame_flipped = cv.flip(frame, 1)

    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+w]

        label, confidence = face_recognizer.predict(faces_roi)
        
        if 0 <= label < len(people):
            label_text = f"{people[label]} ({int(confidence)})"
        else:
            label_text = "Unknown"
        
        # Adjust text position after flip
        flipped_x = frame.shape[1] - (x + w)
        flipped_y = y

        # Draw the rectangle and text on the flipped frame
        cv.rectangle(frame_flipped, (flipped_x, flipped_y), (flipped_x + w, flipped_y + h), (0, 255, 0), thickness=2)
        cv.putText(frame_flipped, label_text, (flipped_x, flipped_y - 10), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
    
    # Display the mirrored frame with correct text
    cv.imshow("Live Face Recognition (Mirror)", frame_flipped)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

