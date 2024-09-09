import cv2
import torch

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = torch.hub.load('ultralytics/yolov5:master', 'yolov5s').to(device)
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

# IoU threshold for considering overlap
iou_threshold = 0.3

# Create a named window with flags
cv2.namedWindow('YOLOv5 Object Detection', cv2.WINDOW_NORMAL)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Skip frames
    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    # Convert the frame to RGB format (required by YOLOv5)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to a tensor and move it to the device
    rgb_tensor = torch.from_numpy(rgb_frame).to(device)  # Ensure tensor is on the correct device
    rgb_tensor = rgb_tensor.permute(2, 0, 1).float()  # Change shape to (C, H, W)
    rgb_tensor /= 255.0  # Normalize to [0, 1]

    # Perform inference
    results = model(rgb_tensor.unsqueeze(0))  # Add batch dimension

    # Initialize lists to store bounding boxes of persons, bicycles, and motorcycles
    persons = []
    bicycles = []
    motorcycles = []

    # Extract bounding boxes for persons, bicycles, and motorcycles
    for result in results.xyxy[0]:
        label = int(result[5].item())  # class label
        confidence = result[4].item()  # confidence
        bbox = result[0:4].int().tolist()  # bounding box
        bbox = [int(coord) for coord in bbox]

        if label == 0:  # person
            persons.append(bbox)
        elif label == 1:  # bicycle
            bicycles.append(bbox)
        elif label == 3:  # motorcycle
            motorcycles.append(bbox)

    # Check for overlaps and draw bounding boxes
    if persons and bicycles + motorcycles:
        # Calculate bounding box that encompasses both person and vehicles
        min_x = min(min(person_bbox[0] for person_bbox in persons), min(v[0] for v in bicycles + motorcycles))
        min_y = min(min(person_bbox[1] for person_bbox in persons), min(v[1] for v in bicycles + motorcycles))
        max_x = max(max(person_bbox[2] for person_bbox in persons), max(v[2] for v in bicycles + motorcycles))
        max_y = max(max(person_bbox[3] for person_bbox in persons), max(v[3] for v in bicycles + motorcycles))

        # Draw a single green box for person and vehicles
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
    else:
        # Draw bounding boxes for persons in red
        for person_bbox in persons:
            cv2.rectangle(frame, (person_bbox[0], person_bbox[1]), (person_bbox[2], person_bbox[3]), (0, 0, 255), 2)

        # Draw bounding boxes for bicycles and motorcycles in green
        for bbox in bicycles + motorcycles:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

    # Display the live feed with a larger window size
    cv2.imshow('YOLOv5 Object Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
