import cv2
import numpy as np

# Load the pre-trained MobileNetSSD model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Define class labels for the objects that the model can detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
           "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 represents the default camera (usually your laptop's built-in camera)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Resize the frame for faster processing
    frame = cv2.resize(frame, (300, 300))

    # Prepare the image for object detection
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # Pass the blob through the network
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by setting a confidence threshold
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            label = CLASSES[class_id]
            score = confidence

            # Get the bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            text = f"{label}: {score:.2f}"
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Object Detection", frame)

    # Exit the loop by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
