import face_recognition
import cv2
import sys

# 1. Load and encode the known face
known_image = face_recognition.load_image_file("Cameron.jpg")
known_encodings_list = face_recognition.face_encodings(known_image)
if len(known_encodings_list) == 0:
    print("Error: No face found in 'Cameron.jpg'. Please use a different image.")
    sys.exit(1)

known_encoding = known_encodings_list[0]
known_names = ["Me"]
known_encodings = [known_encoding]

# 2. Initialize webcam
webcamera = cv2.VideoCapture(0)
if not webcamera.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)

# Optionally set camera resolution
webcamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
webcamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    success, frame = webcamera.read()
    if not success:
        print("Warning: Failed to read from webcam. Exiting...")
        break

    # Convert BGR (OpenCV default) to RGB (face_recognition expects RGB)
    rgb_frame = frame[:, :, ::-1]

    # 3. Detect face locations
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    
    # 4. Encode each face found in the frame
    face_encs = face_recognition.face_encodings(rgb_frame, face_locations)

    # 5. Compare against known encodings
    for (top, right, bottom, left), face_enc in zip(face_locations, face_encs):
        matches = face_recognition.compare_faces(known_encodings, face_enc, tolerance=0.6)
        name = "Unknown"
        
        if True in matches:
            match_idx = matches.index(True)
            name = known_names[match_idx]

        # Draw bounding box and label on the video frame
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # 6. Display the annotated video frame
    cv2.imshow('Webcam', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
webcamera.release()
cv2.destroyAllWindows()
