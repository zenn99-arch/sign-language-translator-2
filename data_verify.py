import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.decomposition import PCA

# Paths to preprocessed data
X_TRAIN_PATH = 'X_train.npy'
X_VAL_PATH = 'X_val.npy'
Y_TRAIN_PATH = 'y_train.npy'
Y_VAL_PATH = 'y_val.npy'
LABEL_ENCODER_PATH = 'label_encoder_classes.npy'

def load_data():
    X_train = np.load(X_TRAIN_PATH)
    X_val = np.load(X_VAL_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    y_val = np.load(Y_VAL_PATH)
    label_encoder_classes = np.load(LABEL_ENCODER_PATH)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = label_encoder_classes
    return X_train, X_val, y_train, y_val, label_encoder

def visualize_keypoints(image_path, keypoints):
    """Visualize keypoints on the image."""
    image = cv2.imread(image_path)
    for i in range(0, len(keypoints), 3):
        x = int(keypoints[i] * image.shape[1])
        y = int(keypoints[i + 1] * image.shape[0])
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    cv2.imshow("Keypoints", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def check_data_quality(X):
    """Check for NaN, Infinite values and consistent shapes."""
    if np.isnan(X).any():
        print("Warning: NaN values found in the data.")
    if np.isinf(X).any():
        print("Warning: Infinite values found in the data.")
    unique_shapes = {x.shape for x in X}
    if len(unique_shapes) > 1:
        print(f"Inconsistent data shapes found: {unique_shapes}")

def check_label_distribution(y):
    """Check distribution of labels."""
    label_count = Counter(y)
    print(f"Label distribution: {label_count}")

def summarize_data(X, y):
    """Print summary statistics of data."""
    print(f"Mean of keypoints: {np.mean(X, axis=0)}")
    print(f"Std of keypoints: {np.std(X, axis=0)}")
    print(f"Label distribution: {Counter(y)}")

def visualize_pca(X, y, label_encoder):
    """Apply PCA and plot data distribution."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, ticks=range(len(label_encoder.classes_)), label='Classes')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA of Keypoints')
    plt.show()

def main():
    X_train, X_val, y_train, y_val, label_encoder = load_data()

    # Check data quality
    print("Checking training data quality...")
    check_data_quality(X_train)
    print("Checking validation data quality...")
    check_data_quality(X_val)

    # Check label distribution
    print("Checking training label distribution...")
    check_label_distribution(y_train)
    print("Checking validation label distribution...")
    check_label_distribution(y_val)

    # Summarize data
    print("Summarizing training data...")
    summarize_data(X_train, y_train)
    print("Summarizing validation data...")
    summarize_data(X_val, y_val)

    # Visualize PCA
    print("Visualizing PCA of training data...")
    visualize_pca(X_train, y_train, label_encoder)

if __name__ == "__main__":
    main()
