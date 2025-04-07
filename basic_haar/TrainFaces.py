import json
import os
import cv2
import numpy as np

# 1. Path to your Haar Cascade file
haar_cascade_path = "haarcascade_frontalface_default.xml"

# 2. Initialize the Haar Cascade
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

def get_images_and_labels(data_folder_path):
    """
    Walks through each person’s folder inside `data_folder_path`,
    reads images, detects faces, and returns:
        - faces: list of cropped face regions
        - labels: list of integer labels corresponding to each face
        - label_dict: mapping of label -> person name
    """
    faces = []
    labels = []
    label_dict = {}

    # This integer will act as our label ID for each person
    current_label_id = 0

    # Loop over each person folder in train_data
    for person_name in os.listdir(data_folder_path):
        person_folder_path = os.path.join(data_folder_path, person_name)
        
        # Skip anything that's not a directory
        if not os.path.isdir(person_folder_path):
            continue

        # Assign a numeric label to this person (e.g., 0 for person1, 1 for person2, etc.)
        label_dict[current_label_id] = person_name

        # Loop over each image in this person's folder
        for image_name in os.listdir(person_folder_path):
            image_path = os.path.join(person_folder_path, image_name)

            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                continue  # skip unreadable or invalid images

            # Convert to grayscale (face recognizers expect grayscale)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image (you might want to tune the scaleFactor & minNeighbors)
            faces_rects = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.3, 
                minNeighbors=5
            )

            # For each face detected, crop it and add to the training set
            for (x, y, w, h) in faces_rects:
                face_roi = gray[y:y+h, x:x+w]
                faces.append(face_roi)
                labels.append(current_label_id)

        current_label_id += 1

    return faces, labels, label_dict


if __name__ == "__main__":
    # Path to the folder containing subfolders of faces
    data_folder_path = "training_data"

    # Get the face ROIs and labels
    faces, labels, label_dict = get_images_and_labels(data_folder_path)

    print(f"Collected {len(faces)} faces from {len(label_dict)} different persons.")

    with open("labels.json", "w") as f:
        json.dump(label_dict, f)

    # 3. Create the LBPH Face Recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # 4. Train the recognizer
    recognizer.train(faces, np.array(labels))

    # 5. Save the trained model to a file
    recognizer.save("training_data.yml")
    print("Training complete. Model saved to 'training_data.yml'.")
