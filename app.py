import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained model
@st.cache_resource
def load_trained_model():
    return load_model("model.h5")

model = load_trained_model()

st.title("🧠 Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0–9) and the model will predict it.")

# File uploader
uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

def preprocess_image(image: Image.Image):
    """
    Preprocess the uploaded image to match MNIST format (28x28, white digit on black background).
    """
    # Convert to grayscale
    image = image.convert("L")
    image = np.array(image)

    # Invert if background is white
    if np.mean(image) > 127:
        image = 255 - image

    # Thresholding
    _, image = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY)

    # Find bounding box of the digit
    coords = cv2.findNonZero(image)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        image = image[y:y+h, x:x+w]

    # Resize to 20x20
    image = cv2.resize(image, (20, 20), interpolation=cv2.INTER_AREA)

    # Pad to 28x28
    padded = np.pad(image, ((4, 4), (4, 4)), "constant", constant_values=0)

    # Normalize
    padded = padded.astype("float32") / 255.0
    padded = padded.reshape(1, 28, 28, 1)

    return padded

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=150)

    # Preprocess
    processed_image = preprocess_image(image)
    st.image(processed_image.reshape(28, 28), caption="Preprocessed (28x28)", width=150)

    # Predict
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)
    st.success(f"Predicted Digit: {predicted_digit}")

    # Show confidence scores
    st.write("Confidence Scores:")
    for i, score in enumerate(prediction[0]):
        st.write(f"{i}: {score:.4f}")
