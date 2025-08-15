import streamlit as st
import os
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import pandas as pd
from io import BytesIO
from sklearn.cluster import DBSCAN
import shutil

# --- Load ResNet50 model for feature extraction ---
@st.cache_resource
def load_model():
    """Loads the ResNet50 model with ImageNet weights."""
    return ResNet50(weights='imagenet', include_top=False, pooling='avg')

model = load_model()

st.title("Image Similarity Clustering App")

# --- Initialize session state for uploaded files ---
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# --- File uploader and duplicate check ---
uploaded = st.file_uploader(
    "Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded:
    existing_file_names = {file.name for file in st.session_state.uploaded_files}
    for file in uploaded:
        if file.name not in existing_file_names:
            st.session_state.uploaded_files.append(file)
            existing_file_names.add(file.name)

# --- Clear All button ---
if st.button("🗑️ Clear All"):
    st.session_state.uploaded_files = []
    if os.path.exists("temp_uploads"):
        shutil.rmtree("temp_uploads")
    st.experimental_rerun()

# --- User input for DBSCAN epsilon value ---
st.markdown("---")
st.subheader("Clustering Settings")
eps_value = st.slider(
    "Select a 'strictness' level (Epsilon): Smaller value means stricter clustering.",
    min_value=0.001,
    max_value=0.2,
    value=0.05,
    step=0.001
)
st.info(f"The current epsilon (eps) value is: {eps_value}. This is the maximum distance for two images to be considered in the same cluster.")

# --- Feature extraction function ---
def extract_features(file_object):
    """Extracts features from an image in memory using the ResNet50 model."""
    try:
        file_object.seek(0)
        img = Image.open(file_object).convert('RGB')
        img = img.resize((224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x, verbose=0)
        return features.flatten()
    except (IOError, OSError) as e:
        st.error(f"Could not process image {file_object.name}. Error: {e}")
        return None

# --- Function to generate cluster name ---
def get_common_cluster_name(filenames):
    if not filenames:
        return "Unknown Cluster"
    if len(filenames) == 1:
        return os.path.splitext(filenames[0])[0]

    names = [os.path.splitext(f.lower().replace('-', ' ').replace('_', ' '))[0].split() for f in filenames]
    common_words = set(names[0])
    for name_parts in names[1:]:
        common_words.intersection_update(name_parts)
    
    generic_terms = {'1', '2', '3d', 'fop', 'bib', 'plunge', 'and', 'hero', 'images', 'image', 'front'}
    meaningful_words = [word for word in common_words if word not in generic_terms]

    if meaningful_words:
        return " ".join(sorted(list(set(meaningful_words))))
    else:
        return os.path.splitext(filenames[0])[0]

# --- Main logic for processing and clustering ---
if st.session_state.uploaded_files:
    with st.spinner('Extracting features and clustering images...'):
        # Extract features
        features = [extract_features(file) for file in st.session_state.uploaded_files]
        valid_features_and_files = [(f, st.session_state.uploaded_files[i]) for i, f in enumerate(features) if f is not None]

        if not valid_features_and_files:
            st.warning("No valid images could be processed.")
            st.stop()

        features = np.array([item[0] for item in valid_features_and_files])
        valid_files = [item[1] for item in valid_features_and_files]

        # Compute similarity and distance
        sim_matrix = cosine_similarity(features)
        sim_matrix = np.clip(sim_matrix, 0.0, 1.0)
        dist_matrix = 1 - sim_matrix

        # DBSCAN clustering
        dbscan = DBSCAN(eps=eps_value, min_samples=1, metric='precomputed')
        labels = dbscan.fit_predict(dist_matrix)

    # Create clusters
    clusters_dict = {}
    for i, label in enumerate(labels):
        clusters_dict.setdefault(label, []).append(i)
    clusters = list(clusters_dict.values())

    # Prepare DataFrame for Excel
    data = []
    for cluster in clusters:
        cluster_files = [valid_files[i].name for i in cluster]
        cluster_name = get_common_cluster_name(cluster_files)
        for fname in cluster_files:
            data.append([cluster_name, os.path.splitext(fname)[0], fname])

    df = pd.DataFrame(data, columns=["Cluster Name", "Image Name (No Ext)", "Exact Filename"])

    # Display clusters
    if not clusters:
        st.info("No clusters found. Try adjusting the 'eps' value or uploading more images.")
    else:
        st.subheader(f"Found {len(clusters)} clusters")
        for cluster in clusters:
            cluster_files = [valid_files[i].name for i in cluster]
            cluster_name = get_common_cluster_name(cluster_files)
            with st.expander(f"Cluster: {cluster_name} ({len(cluster)} images)", expanded=True):
                cols = st.columns(len(cluster_files))
                for col, idx in zip(cols, cluster):
                    col.image(valid_files[idx], caption=valid_files[idx].name, use_container_width=True)

    # Download Excel
    excel_buffer = BytesIO()
    df.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)
    st.download_button(
        label="📥 Download Clusters Excel",
        data=excel_buffer,
        file_name="image_clusters.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
