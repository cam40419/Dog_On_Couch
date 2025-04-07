import cv2
import numpy as np
import tensorflow as tf
import keras
import sys
print("Python executable:", sys.executable)

print(keras.__version__)
print(tf.__version__)



# Load the pre-trained FaceNet model
model = keras.saving.load_model('facenet\\facenet_keras.h5')
print("Model Loaded Successfully")

def preprocess_image(image_path):
    # Load the image using OpenCV
    img = cv2.imread(image_path)

    # Convert the image to RGB (FaceNet expects RGB images)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the image to 160x160 pixels
    img = cv2.resize(img, (160, 160))

    # Normalize the pixel values
    img = img.astype('float32') / 255.0

    # Expand dimensions to match the input shape of FaceNet (1, 160, 160, 3)
    img = np.expand_dims(img, axis=0)

    return img

def get_face_embedding(model, image_path):
    # Preprocess the image
    img = preprocess_image(image_path)

    # Generate the embedding
    embedding = model.predict(img)

    return embedding

from numpy import linalg as LA

def compare_faces(embedding1, embedding2, threshold=0.5):
    # Compute the Euclidean distance between the embeddings
    distance = LA.norm(embedding1 - embedding2)

    # Compare the distance to the threshold
    if distance < threshold:
        print("Face Matched.")
    else:
        print("Faces are different.")

    return distance