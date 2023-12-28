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

    # Perform inference
    results = model(rgb_frame)

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
    for person_bbox in persons:
        for vehicle_bbox in bicycles + motorcycles:
            if (
                person_bbox[0] < vehicle_bbox[2] and
                person_bbox[2] > vehicle_bbox[0] and
                person_bbox[1] < vehicle_bbox[3] and
                person_bbox[3] > vehicle_bbox[1]
            ):
                # Combine person and bicycle/motorcycle bounding boxes and draw a green box
                combined_bbox = [
                    min(person_bbox[0], vehicle_bbox[0]),
                    min(person_bbox[1], vehicle_bbox[1]),
                    max(person_bbox[2], vehicle_bbox[2]),
                    max(person_bbox[3], vehicle_bbox[3]),
                ]
                cv2.rectangle(frame, (combined_bbox[0], combined_bbox[1]), (combined_bbox[2], combined_bbox[3]), (0, 255, 0), 2)
                break  # No need to check further

        else:
            # Draw a red box if there is no overlap with any bicycle or motorcycle
            cv2.rectangle(frame, (person_bbox[0], person_bbox[1]), (person_bbox[2], person_bbox[3]), (0, 0, 255), 2)

    # Draw bounding boxes for bicycles and motorcycles
    for bbox in bicycles + motorcycles:
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    # Display the live feed with a larger window size
    cv2.imshow('YOLOv5 Object Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
