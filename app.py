import streamlit as st
import os
import zipfile
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import pandas as pd
from io import BytesIO

# Load ResNet50 model for feature extraction
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

st.title("Image Similarity Clustering App")

uploaded_files = st.file_uploader(
    "Upload Images or ZIP file", type=["jpg", "jpeg", "png", "zip"], accept_multiple_files=True
)

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

def get_common_cluster_name(filenames):
    names = [os.path.splitext(f)[0] for f in filenames]
    if len(names) == 1:
        return names[0]
    split_names = [name.split() for name in names]
    common = set(split_names[0])
    for parts in split_names[1:]:
        common &= set(parts)
    common_name = " ".join([w for w in names[0].split() if w in common])
    return common_name if common_name else names[0]

def transitive_clustering(sim_matrix, min_threshold=0.93, max_threshold=0.99):
    n = sim_matrix.shape[0]
    clusters = []
    visited = set()

    for idx in range(n):
        if idx in visited:
            continue
        cluster = set([idx])
        queue = [idx]
        visited.add(idx)

        while queue:
            current = queue.pop(0)
            for j in range(n):
                if j not in visited and min_threshold <= sim_matrix[current, j] <= max_threshold:
                    cluster.add(j)
                    queue.append(j)
                    visited.add(j)

        clusters.append(list(cluster))
    return clusters

if uploaded_files:
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_paths = []

    # Process uploaded files
    for file in uploaded_files:
        if file.name.endswith(".zip"):
            # Extract ZIP
            with zipfile.ZipFile(file, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
            # Add all images from ZIP to file_paths
            for root, _, files in o_
