import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

#from ultralytics import YOLO

#model = YOLO("yolov8n-cls.pt")

#results = model.train(data=r"C:\Users\Roshan\OneDrive\Desktop\opencv_proj\opencv_attendence\yolo_v9\datasets_yolov8", epochs=50, imgsz=640)
import torch

print("PyTorch version:", torch.__version__)
print("Is CUDA available?", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))

    tensor = torch.rand(3, 3).to("cuda")
    print("Tensor on GPU:", tensor)
else:
    print("GPU not detected by PyTorch.")

from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO("yolov8n-cls.pt")

    results = model.train(
        data=r"C:\Users\Roshan\OneDrive\Desktop\opencv_proj\opencv_attendence\yolo_v9\datasets_yolov8",
        epochs=50,
        imgsz=620,
        batch = 10,
        device="cuda"  
    )

