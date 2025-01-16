import cv2
import numpy as np

def load_liveness_model(model_path):
    """Load the pre-trained Caffe model for liveness detection with FP16 precision."""
    model = cv2.dnn.readNetFromCaffe(model_path + ".prototxt", model_path + ".caffemodel")
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU_FP16)
    print("Caffe model loaded successfully.")
    return model

def preprocess_frame(frame):
    """Preprocess the camera frame for prediction."""
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 117, 123), swapRB=False, crop=False)
    return blob


def predict_liveness(model, blob):
    """Predict whether the frame is live or spoofed."""
    model.setInput(blob)
    prediction = model.forward()
    print("Prediction output shape:", prediction.shape)  # Debugging

    # Extract confidence scores and bounding boxes
    detections = prediction[0, 0, :, :]
    confidences = detections[:, 2]  # Confidence scores
    bbox_sizes = (detections[:, 5] - detections[:, 3]) * (detections[:, 6] - detections[:, 4])  # Bounding box sizes

    # Filter detections with a minimum confidence threshold and valid bounding box size
    valid_detections = [(conf, bbox) for conf, bbox in zip(confidences, bbox_sizes) if conf > 0.5 and bbox > 0.1]

    if valid_detections:
        # Sort by confidence score and select the highest
        valid_detections.sort(key=lambda x: x[0], reverse=True)
        max_confidence, max_bbox = valid_detections[0]
        print(f"Max confidence: {max_confidence}, Max bounding box size: {max_bbox}")
        return "Live" if max_confidence > 0.7 else "Spoof"  # Adjusted threshold
    else:
        return "Spoof"

def main():
    """Main function to perform real-time liveness detection."""
    model_path = "models/liveness_model"  # Static path to the Caffe model

    print("Loading model...")
    model = load_liveness_model(model_path)

    print("Starting camera...")
    cap = cv2.VideoCapture(2)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Preprocess the frame
        blob = preprocess_frame(frame)

        # Predict liveness
        result = predict_liveness(model, blob)

        # Display the result on the frame
        label = f"Result: {result}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Liveness Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
