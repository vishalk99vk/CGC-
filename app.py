import streamlit as st
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os
import glob

# --- Configuration ---
# Directory where your images are stored
IMAGE_DIR = "uploaded_images" # Change this if your images are in a different folder
NUM_CLUSTERS = 3 # You can adjust this number based on your dataset

# --- Function to load and preprocess images ---
@st.cache_resource
def load_and_preprocess_images(image_paths):
    """
    Loads images, resizes them, and converts them to tensors.
    """
    image_tensors = []
    image_names = []
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert("RGB")
            tensor = transform(image)
            image_tensors.append(tensor)
            image_names.append(os.path.basename(img_path))
        except Exception as e:
            st.error(f"Error loading image {img_path}: {e}")
            
    return torch.stack(image_tensors), image_names

# --- Function to extract features using a pre-trained CNN ---
@st.cache_resource
def extract_features(images):
    """
    Extracts features from images using a pre-trained VGG16 model.
    """
    # Load a pre-trained model
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    # Remove the classifier layers to get a feature extractor
    feature_extractor = torch.nn.Sequential(*list(model.features.children()))
    
    # Put the model in evaluation mode
    feature_extractor.eval()
    
    with torch.no_grad():
        features = feature_extractor(images)
    
    # Flatten the features for each image
    features = features.view(images.size(0), -1)
    
    return features.numpy()

# --- Main Streamlit App ---
def main():
    st.title("Image Clustering for Similar Products")
    st.write("This application clusters images of products based on their visual similarity.")

    # Find all JPEG images in the specified directory
    image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    if not image_paths:
        st.warning(f"No JPEG images found in the '{IMAGE_DIR}' directory. Please make sure your images are there.")
        return

    st.write(f"Found {len(image_paths)} images to process.")

    with st.spinner("Loading and preprocessing images..."):
        image_tensors, image_names = load_and_preprocess_images(image_paths)

    with st.spinner("Extracting features from images..."):
        features = extract_features(image_tensors)
        st.success("Feature extraction complete.")

    # Perform clustering
    st.subheader(f"Clustering with K-Means (k={NUM_CLUSTERS})")
    with st.spinner(f"Clustering images into {NUM_CLUSTERS} groups..."):
        kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features)
        st.success("Clustering complete.")

    st.markdown("---")

    # Display clusters
    st.subheader("Image Clusters")
    for cluster_id in range(NUM_CLUSTERS):
        st.markdown(f"### Cluster {cluster_id + 1}")
        
        # Get the indices of images in the current cluster
        cluster_indices = np.where(clusters == cluster_id)[0]
        
        if len(cluster_indices) == 0:
            st.info("This cluster is empty.")
            continue
            
        # Create a list of images for this cluster
        cluster_images = [Image.open(image_paths[i]) for i in cluster_indices]
        cluster_labels = [image_names[i] for i in cluster_indices]
        
        # Display images in a grid
        cols = st.columns(min(len(cluster_images), 5)) # Display up to 5 images per row
        
        for i, (col, img, label) in enumerate(zip(cols, cluster_images, cluster_labels)):
            with col:
                st.image(img, caption=label, use_column_width=True)

if __name__ == "__main__":
    # Create the image directory if it doesn't exist
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        st.warning(f"Created directory '{IMAGE_DIR}'. Please place your images here and restart the app.")
    else:
        main()
