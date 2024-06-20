import cv2
from ultralytics import YOLO
import os

# Load the model
model = YOLO('/home/codezeros/Documents/fire&smoke detection/Test/best.pt')
# Open the video file

video_path = "/home/codezeros/Documents/fire&smoke detection/Test/s1_vid.mp4"
cap = cv2.VideoCapture(video_path)

# Create a folder to save detected frames if it doesn't exist
# output_folder = "detected_frames"
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Perform inference on the frame
    results = model(frame)
    label = "smoke"
    # Draw bounding boxes on the frame
    for result in results:
        # if result.boxes.xyxy:
            print(result)
            box = result.boxes.xyxy
            box = box.tolist()
            conf = result.boxes.conf[0]  # Confidence score
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 5)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"{label}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Save the detected frame
        # cv2.imwrite(os.path.join(output_folder, f"detected_frame_{conf:.2f}.jpg"), frame)
    # Display the frame with bounding boxes
    cv2.imshow('Video Inference', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




