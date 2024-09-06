import numpy as np
import imgaug.augmenters as iaa
import os
from tqdm import tqdm

# Define your augmentations
seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-25, 25)),
    iaa.Multiply((0.8, 1.2)),
    iaa.GaussianBlur(sigma=(0, 1.0))
])

def load_preprocessed_data():
    """Load the preprocessed training data."""
    if not os.path.exists('X_train.npy') or not os.path.exists('y_train.npy'):
        raise FileNotFoundError("Preprocessed data files not found.")
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    return X_train, y_train

def infer_image_dimensions(sample_image):
    """Infer image dimensions from a single sample image."""
    if sample_image.ndim == 2:
        # Grayscale image: height x width
        height, width = sample_image.shape
        return height, width, 1
    elif sample_image.ndim == 3:
        # RGB image: height x width x channels
        height, width, channels = sample_image.shape
        if channels == 1:
            return height, width, 1
        elif channels == 3:
            return height, width, 3
        else:
            raise ValueError(f"Unexpected number of channels: {channels}")
    else:
        print(f"Unsupported image format: {sample_image.ndim} dimensions")
        raise ValueError("Unsupported image data format.")

def augment_data(X, y, num_augmentations=2):
    """Perform data augmentation."""
    if len(X) == 0:
        raise ValueError("No data to augment.")
    
    # Get image dimensions from the first sample
    sample_image = X[0].reshape((-1, 1))  # Temporarily reshape to access dimensions
    try:
        height, width, channels = infer_image_dimensions(sample_image)
    except ValueError as e:
        print(e)
        return np.array([]), np.array([])  # Return empty arrays in case of error
    
    augmented_X = []
    augmented_y = []
    
    for img, label in tqdm(zip(X, y), total=len(X), desc="Augmenting Data"):
        img = img.reshape((height, width, channels)) if img.ndim == 1 else img
        if img.shape != (height, width, channels):
            print(f"Warning: Image shape mismatch. Expected: ({height}, {width}, {channels}), Got: {img.shape}")
            continue
        
        # Convert image to uint8 if necessary
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        augmented_images = seq(images=[img] * num_augmentations)
        for augmented_img in augmented_images:
            augmented_X.append(augmented_img.flatten())
            augmented_y.append(label)
    
    return np.array(augmented_X), np.array(augmented_y)

def save_augmented_data(X, y):
    """Save the augmented data."""
    np.save('X_train_augmented.npy', X)
    np.save('y_train_augmented.npy', y)
    print("Augmented data saved.")

def main():
    """Main function to execute the augmentation process."""
    X_train, y_train = load_preprocessed_data()
    
    augmented_X_train, augmented_y_train = augment_data(X_train, y_train)
    if augmented_X_train.size > 0 and augmented_y_train.size > 0:
        save_augmented_data(augmented_X_train, augmented_y_train)
    else:
        print("No augmented data to save.")

if __name__ == "__main__":
    main()
