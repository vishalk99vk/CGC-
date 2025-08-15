import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from io import BytesIO
import pandas as pd

st.title("High-Performance Image Similarity Clustering")

# --- Load ResNet50 model ---
@st.cache_resource
def load_model():
    return ResNet50(weights='imagenet', include_top=False, pooling='avg')
model = load_model()

# --- Upload images ---
uploaded = st.file_uploader("Upload Images (up to 500)", type=["jpg","jpeg","png"], accept_multiple_files=True)
if not uploaded:
    st.stop()

# --- Feature extraction functions ---
def extract_color_histogram(img, bins=32):
    img = img.resize((224,224))
    arr = np.array(img)
    hist_r = np.histogram(arr[:,:,0], bins=bins, range=(0,255))[0]
    hist_g = np.histogram(arr[:,:,1], bins=bins, range=(0,255))[0]
    hist_b = np.histogram(arr[:,:,2], bins=bins, range=(0,255))[0]
    hist = np.concatenate([hist_r,hist_g,hist_b])
    return hist / np.sum(hist)

def extract_texture_features(img):
    img_gray = img.convert('L').resize((224,224))
    arr = np.array(img_gray)
    # LBP feature
    lbp = local_binary_pattern(arr, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), density=True)
    # GLCM features
    glcm = graycomatrix(arr, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0,0]
    correlation = graycoprops(glcm, 'correlation')[0,0]
    energy = graycoprops(glcm, 'energy')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    return np.concatenate([lbp_hist, [contrast, correlation, energy, homogeneity]])

def extract_deep_features(img):
    img = img.resize((224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feat = model.predict(x, verbose=0)
    return feat.flatten()

def extract_combined_features(img):
    color_feat = extract_color_histogram(img)
    texture_feat = extract_texture_features(img)
    deep_feat = extract_deep_features(img)
    combined = np.concatenate([color_feat, texture_feat, deep_feat])
    return combined

# --- Extract features for all images ---
st.info("Extracting features... This may take a few minutes for 500 images.")
features = []
valid_files = []
for file in uploaded:
    try:
        img = Image.open(file).convert('RGB')
        feat = extract_combined_features(img)
        features.append(feat)
        valid_files.append(file)
    except Exception as e:
        st.warning(f"Skipping file {file.name}: {e}")

features = np.array(features)
features = StandardScaler().fit_transform(features)

# --- Compute similarity and clustering ---
st.info("Computing similarity and clustering...")
sim_matrix = cosine_similarity(features)
dist_matrix = 1 - sim_matrix  # distance
eps_value = 0.05  # ~95% similarity threshold
db = DBSCAN(eps=eps_value, min_samples=1, metric='precomputed')
labels = db.fit_predict(dist_matrix)

# --- Prepare clusters ---
clusters_dict = {}
for idx, label in enumerate(labels):
    clusters_dict.setdefault(label, []).append(valid_files[idx].name)

# --- Display clusters ---
st.subheader(f"Found {len(clusters_dict)} clusters")
for label, files in clusters_dict.items():
    with st.expander(f"Cluster {label} ({len(files)} images)", expanded=True):
        cols = st.columns(min(5, len(files)))
        for col, fname in zip(cols, files):
            file_obj = next(f for f in valid_files if f.name == fname)
            col.image(Image.open(file_obj), caption=fname, use_container_width=True)

# --- Download Excel ---
data = []
for label, files in clusters_dict.items():
    for f in files:
        data.append([label, f])
df = pd.DataFrame(data, columns=["Cluster", "Filename"])
excel_buffer = BytesIO()
df.to_excel(excel_buffer, index=False)
st.download_button("Download Clusters Excel", data=excel_buffer, file_name="clusters.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
