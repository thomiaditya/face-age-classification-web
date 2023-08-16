import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from mtcnn import MTCNN
import cv2
import os
import requests

from keras import backend as K

MODEL_NAME = 'earnest-wood-70'

def f1_score(y_true, y_pred):
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    if c3 == 0:
        return 0.0

    epsilon = K.epsilon()
    precision = c1 / (c2 + epsilon)
    recall = c1 / (c3 + epsilon)

    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1_score

MODEL_URL = "https://github.com/thomiaditya/face-age-classification-web/releases/download/1.0/earnest-wood-70.h5"
MODEL_PATH = f"{MODEL_NAME}.h5"

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        # If model doesn't exist, download it
        with st.spinner("Downloading model..."):
            response = requests.get(MODEL_URL, allow_redirects=True)
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
    
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"f1_score": f1_score})
    return model

# Load the trained model
# Print loading model message to streamlit app
placeholder = st.empty()

placeholder.text("Loading model...")
model = load_model()
# Clear loading model message
placeholder.text(f"Loaded model `{MODEL_NAME}`!")

detector = MTCNN()
class_names = ['anak-anak', 'dewasa', 'remaja']

def detect_and_crop_face(img):
    # Convert PIL image to numpy array
    img_array = np.array(img)
    img_array = img_array[:, :, :3]

    # Detect faces in the image
    faces = detector.detect_faces(img_array)
    if len(faces) == 0:
        return None  # No face detected

    # Take the first face detected (in case of multiple faces)
    x, y, width, height = faces[0]['box']

    # Crop the face
    cropped_face = img_array[y:y+height, x:x+width]

    return cropped_face

def predict_image(img):
    img = Image.fromarray(img)
    img = img.resize((200, 200))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    return np.argmax(predictions)

st.title('Face Age Classification menggunakan Fine tuning Pretrained VGGFace model')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    cropped_face = detect_and_crop_face(image)
    if cropped_face is not None:
        st.image(cropped_face, caption='Cropped Face.', use_column_width=True)

        if st.button('Predict'):
            label = predict_image(cropped_face)
            # Show the class label in big and bold and uppercase with green color
            st.title(class_names[label].upper())
    else:
        st.write("No face detected!")