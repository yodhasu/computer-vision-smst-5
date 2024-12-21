import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
import pickle

def load_images_from_folder(folder):
    images = []
    labels = []
    file_names = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for filename in file_names:
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(filename.split('_')[0])
    return images, labels

def preprocess_images(images, size=(224, 224)):
    processed_images = []
    for img in images:
        resized_img = cv2.resize(img, size)
        blurred_img = cv2.GaussianBlur(resized_img, (5, 5), 0)
        gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        normalized_img = cv2.normalize(sobel_magnitude, None, 0, 1, cv2.NORM_MINMAX)
        normalized_img_3channel = np.stack([normalized_img] * 3, axis=-1)
        processed_images.append(normalized_img_3channel)
    return np.array(processed_images)

def extract_conference_features(images, model):
    features = []
    preprocessed_images = preprocess_images(images)
    for img in preprocessed_images:
        preprocessed_img = np.expand_dims(img, axis=0)
        feature = model.predict(preprocessed_img)
        features.append(feature.flatten())
    return np.array(features)

def extract_orb_features(images):
    preprocessed_images = preprocess_images(images)
    orb = cv2.ORB_create()
    descriptors = []
    for img in preprocessed_images:
        img_uint8 = (img * 255).astype(np.uint8)
        gray_img = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        _, des = orb.detectAndCompute(gray_img, None)
        descriptors.append(des)
    return descriptors

def save_features(global_features, global_labels, local_features, save_dir="features_cache_5"):
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, "global_features.npz"), features=global_features, labels=global_labels)
    with open(os.path.join(save_dir, "local_features.pkl"), "wb") as f:
        pickle.dump(local_features, f)

def load_features(save_dir="features_cache_5"):
    global_data = np.load(os.path.join(save_dir, "global_features.npz"))
    global_features, global_labels = global_data["features"], global_data["labels"]
    with open(os.path.join(save_dir, "local_features.pkl"), "rb") as f:
        local_features = pickle.load(f)
    return global_features, global_labels, local_features

def global_matching(query_feature, database_features, database_labels, top_k=5):
    similarities = cosine_similarity([query_feature], database_features)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [(database_labels[i], similarities[i]) for i in top_indices]

def local_matching(query_descriptors, candidate_descriptors):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_per_candidate = []
    for des in candidate_descriptors:
        if des is None or query_descriptors is None:
            matches_per_candidate.append(0)
            continue
        matches = bf.match(query_descriptors, des)
        matches_per_candidate.append(len(matches))
    best_match_idx = np.argmax(matches_per_candidate)
    return best_match_idx, matches_per_candidate

def live_detection(model, global_features, local_features, labels, frame):
    orb = cv2.ORB_create()

    # Preprocess the frame
    processed_frame = preprocess_images([frame])[0]
    processed_frame_uint8 = (processed_frame * 255).astype(np.uint8)

    # Extract global feature
    query_global_feature = model.predict(np.expand_dims(processed_frame, axis=0)).flatten()

    # Convert to grayscale and extract local features
    frame_gray = cv2.cvtColor(processed_frame_uint8, cv2.COLOR_RGB2GRAY)
    _, query_local_descriptors = orb.detectAndCompute(frame_gray, None)

    # Global feature matching
    top_candidates = global_matching(query_global_feature, global_features, labels, top_k=5)

    # Extract candidate descriptors
    candidate_images_indices = [np.where(labels == label)[0][0] for label, _ in top_candidates]
    candidate_descriptors = [local_features[idx] for idx in candidate_images_indices]

    # Local feature matching
    best_idx, matches_per_candidate = local_matching(query_local_descriptors, candidate_descriptors)

    # Get best match label
    best_match_label = labels[candidate_images_indices[best_idx]]

    # Annotate the frame
    cv2.putText(frame, f"Detected: {best_match_label}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame, best_match_label