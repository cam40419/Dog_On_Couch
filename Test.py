import cv2
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
print(model.names)
webcamera = cv2.VideoCapture(0)
# webcamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# webcamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    success, frame = webcamera.read()
    
    # If for some reason the camera feed isn't available, break
    if not success:
        print("Failed to read from webcam. Exiting...")
        break

    # Perform detection or tracking
    results = model.track(
        source=frame,   # frame to run inference on
        conf=0.6,       # confidence threshold
        imgsz=480       # image size for inference
    )

    # results is a list of 'Results' objects, one per image/frame.
    # In a real-time loop, it's typically just 1 item (results[0]).
    # Each results[0].boxes has info about bounding boxes, conf, class, etc.

    # You can get an annotated image (with bounding boxes and labels) like this:
    annotated_frame = results[0].plot()

    # Draw additional text (e.g., number of detections) on the annotated frame:
    cv2.putText(
        annotated_frame, 
        f"Total: {len(results[0].boxes)}", 
        (50, 50), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 0, 255), 
        2, 
        cv2.LINE_AA
    )

    # Show the annotated frame
    cv2.imshow("Live Camera", annotated_frame)

    # If the user presses 'q', exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
webcamera.release()
cv2.destroyAllWindows()
