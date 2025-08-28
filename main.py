import os
import pickle
import streamlit as st
import numpy as np
import tensorflow as tf

from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors

st.title("Apparel Suggestion System")

# --- Load embeddings and filenames ---
try:
    feature_list = pickle.load(open("ëmbeddings.pkl", "rb"))   # fixed filename
    filenames = pickle.load(open("filenames.pkl", "rb"))
except FileNotFoundError as e:
    st.error(f"Required data file missing: {e}")
    st.stop()

# --- Build model ---
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])

# --- Utility functions ---
def save_uploaded_file(uploaded_file) -> str | None:
    """
    Saves the uploaded file to ./uploads and returns the absolute path.
    """
    try:
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)

        # Sanitize filename
        raw_name = os.path.basename(uploaded_file.name)
        safe_name = raw_name.replace(" ", "_")
        file_path = os.path.join(upload_dir, safe_name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def feature_extraction(img_path: str, model) -> np.ndarray:
    """
    Loads image, preprocesses, and extracts a pooled feature vector.
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array, verbose=0).flatten()
    features = features / (norm(features) + 1e-10)  # Normalize
    return features

def recommender(features, features_list):
    """
    Finds the top-6 similar items using Euclidean distance.
    """
    neighbors = NearestNeighbors(n_neighbors=6, algorithm="brute", metric="euclidean")
    neighbors.fit(features_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# --- App flow ---
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
if uploaded_file is not None:
    saved_path = save_uploaded_file(uploaded_file)
    if saved_path is None:
        st.stop()

    st.success("File uploaded successfully!")
    st.image(saved_path, caption="Uploaded Image", use_container_width=True)

    try:
        # Extract features
        features = feature_extraction(saved_path, model)
        st.write("Feature vector shape:", features.shape)

        # Get recommendations
        indices = recommender(features, feature_list)

        # Display recommended images
        cols = st.columns(5)
        for i, col in enumerate(cols):
            if i < len(indices[0]) - 1:  # Skip the first as it's the same image
                with col:
                    st.image(filenames[indices[0][i + 1]])
    except FileNotFoundError:
        st.error(f"File not found at path: {saved_path}. Please re-upload.")
    except Exception as e:
        st.error(f"Error during feature extraction: {e}")
else:
    st.info("Upload an image to extract features.")
