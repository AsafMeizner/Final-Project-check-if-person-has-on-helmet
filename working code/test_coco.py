# Step 3: Create a Python script for object detection
import cv2
import torch
from pathlib import Path

# Load YOLOv5 model with pre-trained weights
model = torch.hub.load('ultralytics/yolov5:master', 'yolov5s', pretrained=True)

# Set the model to evaluation mode
model = model.eval()

# Open a connection to the webcam (change the index if you have multiple cameras)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Perform inference
    results = model(frame)

    # Display the results
    cv2.imshow('YOLOv5 Object Detection', results.render()[0])

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
