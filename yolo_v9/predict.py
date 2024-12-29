from ultralytics import YOLO
import numpy as np

model = YOLO(r"C:\Users\Roshan\runs\classify\train16\weights\best.pt")

results = model(r"C:\Users\Roshan\OneDrive\Pictures\covershot\IMG-20241126-WA0035.jpg")

name_dict = results[0].names
probs = results[0].probs.data 

if not isinstance(probs, np.ndarray):
    probs = probs.cpu().numpy()
print(name_dict)
print(probs)
top_class = name_dict[np.argmax(probs)]
print(f"The class with the highest probability is: {top_class}")
