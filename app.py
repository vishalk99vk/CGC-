import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# --- Parameters ---
folder_path = "path_to_your_images"
radius = 3
n_points = 8 * radius
eps = 0.05  # DBSCAN distance threshold (tune for ~95% similarity)
min_samples = 2  # minimum images per cluster

# --- Helper functions ---
def extract_texture(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points+3), density=True)
    return hist

def extract_color(image):
    # Mean RGB values
    return cv2.mean(image)[:3]

def extract_test_level(filename):
    # Example: extract numeric from filename like "product_12mg.jpg"
    import re
    match = re.search(r'(\d+)', filename)
    return [float(match.group(1))] if match else [0]

# --- Load images and extract features ---
features = []
image_names = []

for fname in os.listdir(folder_path):
    if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(folder_path, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        tex = extract_texture(img)
        col = extract_color(img)
        test = extract_test_level(fname)
        feat = np.concatenate([tex, col, test])
        features.append(feat)
        image_names.append(fname)

features = np.array(features)
features = StandardScaler().fit_transform(features)  # normalize features

# --- DBSCAN clustering ---
db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
labels = db.fit_predict(features)

# --- Output clusters ---
clusters = {}
for label, fname in zip(labels, image_names):
    clusters.setdefault(label, []).append(fname)

for cluster_id, imgs in clusters.items():
    print(f"Cluster {cluster_id}: {len(imgs)} images")
    print(imgs)
    print("------")
