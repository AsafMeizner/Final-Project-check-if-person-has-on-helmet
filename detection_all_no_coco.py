import cv2
import numpy as np
import tensorflow as tf

# Load YOLO models
def load_yolo_model(model_path):
    return tf.saved_model.load(model_path)

# Perform object detection using the YOLO model
def detect_objects(model, image):
    input_tensor = tf.convert_to_tensor(image)
    detections = model(input_tensor)
    return detections

# Draw bounding boxes on the image based on detection results
def draw_boxes(image, boxes, class_ids, colors):
    for i in range(len(boxes)):
        box = boxes[i]
        class_id = int(class_ids[i])
        color = colors[class_id]
        cv2.rectangle(image, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), color, 2)

# Check for overlap between two bounding boxes
def check_overlap(box1, box2):
    overlap_x = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    overlap_y = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    overlap_area = overlap_x * overlap_y
    area_box1 = (box1[3] - box1[1]) * (box1[2] - box1[0])
    area_box2 = (box2[3] - box2[1]) * (box2[2] - box2[0])
    overlap_ratio = overlap_area / min(area_box1, area_box2)
    return overlap_ratio > 0.5  # Adjust the threshold as needed

# Main function for real-time object detection
def main():
    # Load YOLO models
    person_model = load_yolo_model("path/to/person/model")
    bicycle_model = load_yolo_model("path/to/bicycle/model")
    helmet_model = load_yolo_model("path/to/helmet/model")

    # Open webcam
    cap = cv2.VideoCapture(0)

    # Colors for bounding boxes (BGR format)
    person_color = (0, 255, 0)  # Green
    no_helmet_color = (0, 0, 255)  # Red

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect people
        person_detections = detect_objects(person_model, frame)

        # Detect bicycles
        bicycle_detections = detect_objects(bicycle_model, frame)

        # Detect helmets
        helmet_detections = detect_objects(helmet_model, frame)

        # Check for overlap between people and bicycles
        for person_box in person_detections['detection_boxes']:
            for bicycle_box in bicycle_detections['detection_boxes']:
                if check_overlap(person_box, bicycle_box):
                    # Check for helmets near the person's head
                    for helmet_box in helmet_detections['detection_boxes']:
                        if check_overlap(person_box, helmet_box):
                            draw_boxes(frame, [person_box], [0], person_color)
                        else:
                            draw_boxes(frame, [person_box], [0], no_helmet_color)

        # Display the frame
        cv2.imshow('Object Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
