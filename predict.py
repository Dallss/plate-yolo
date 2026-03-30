import random
from pathlib import Path
from ultralytics import YOLO

# Path to test images folder
test_folder = Path("test/images")

# Pick a random image
test_image = random.choice(list(test_folder.glob("*.*")))  # matches any file
print(f"Using image: {test_image}")

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Run prediction
results = model.predict(source=str(test_image), conf=0.05)  # lower threshold if needed

# Visualize/save results
results[0].show()  # shows image in window (if supported)

# Log bounding boxes
boxes = results[0].boxes
for i, box in enumerate(boxes):
    xyxy = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
    cls = int(box.cls[0].cpu().numpy())  # class index
    conf = float(box.conf[0].cpu().numpy())  # confidence score
    print(f"Box {i}: {xyxy}, class={cls}, conf={conf:.2f}")