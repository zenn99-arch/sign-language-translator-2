import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import logging
import argparse

def load_data(X_train_path, X_val_path, y_train_path, y_val_path, label_encoder_path):
    """Load and preprocess the data."""
    X_train = np.load(X_train_path)
    X_val = np.load(X_val_path)
    y_train = np.load(y_train_path)
    y_val = np.load(y_val_path)
    label_classes = np.load(label_encoder_path)

    # Normalize data
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    return X_train, X_val, y_train, y_val, label_classes

def create_model(input_shape, num_classes):
    """Build and compile the neural network model."""
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_history(history):
    """Plot training and validation history."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

def main(args):
    """Main function for training the model."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info('Loading data...')
    X_train, X_val, y_train, y_val, label_classes = load_data(
        args.X_train_path, args.X_val_path, args.y_train_path, args.y_val_path, args.label_encoder_path
    )

    num_classes = len(label_classes)
    
    logging.info('Creating model...')
    model = create_model(X_train.shape[1], num_classes)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint('best_model.keras', save_best_only=True, verbose=1)
    ]

    logging.info('Training model...')
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    model.save('final_model.keras')
    logging.info('Model saved to final_model.keras')

    plot_history(history)
    np.save('training_history.npy', history.history)
    logging.info('Training history saved to training_history.npy')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a sign language model.')
    parser.add_argument('--X_train_path', type=str, required=True, help='Path to the training features.')
    parser.add_argument('--X_val_path', type=str, required=True, help='Path to the validation features.')
    parser.add_argument('--y_train_path', type=str, required=True, help='Path to the training labels.')
    parser.add_argument('--y_val_path', type=str, required=True, help='Path to the validation labels.')
    parser.add_argument('--label_encoder_path', type=str, required=True, help='Path to the label encoder classes.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    
    args = parser.parse_args()
    main(args)
