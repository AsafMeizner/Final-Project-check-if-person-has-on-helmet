import cv2
import torch

# Model
model = torch.hub.load('ultralytics/yolov5:master', 'yolov5s')
model.classes = [0, 1, 3]  # person, bicycle, and motorcycle

# Set the model to evaluation mode
model = model.eval()

# Open a connection to the webcam (change the index if you have multiple cameras)
cap = cv2.VideoCapture(0)

# Set inference size
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Set skip frames
skip_frames = 5
frame_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Skip frames
    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    # Convert the frame to RGB format (required by YOLOv5)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(rgb_frame)

    # Display the results on the live feed
    for result in results.xyxy[0]:
        label = int(result[5].item())  # class label
        confidence = result[4].item()  # confidence
        if label in model.classes and confidence > 0.5:
            bbox = result[0:4].int().tolist()  # bounding box
            bbox = [int(coord) for coord in bbox]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, f'{model.names[label]} {confidence:.2f}', (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the live feed
    cv2.imshow('YOLOv5 Object Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()