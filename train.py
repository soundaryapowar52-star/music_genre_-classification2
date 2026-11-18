"""
train.py
--------
Trains a 2D Convolutional Neural Network (CNN) in Keras to classify music genres.

Key Features & Fixes:
1. Universal Keras 2 & Keras 3 compatibility using the Functional API (avoids KerasTensor bugs).
2. Fast data loading from compressed .npz format (with .json fallback).
3. 80/20 Stratified train/validation split.
4. Robust regularization (Batch Normalization, Dropout, L2 penalty).
5. Comprehensive metrics: Loss/Accuracy curves, Confusion Matrix, and Classification Report.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers

# Paths & Hyperparameters
DATA_NPZ = "data/data_mfcc.npz"
DATA_JSON = "data/data_mfcc.json"
MODEL_DIR = "models"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "music_genre_cnn.keras")
MAPPING_SAVE_PATH = os.path.join(MODEL_DIR, "genre_mapping.json")
PLOTS_DIR = "plots"

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.0001


def load_dataset(npz_path=DATA_NPZ, json_path=DATA_JSON):
    """
    Loads preprocessed MFCC features and labels.
    Prefers fast .npz format, falls back to .json if needed.
    """
    if os.path.exists(npz_path):
        print(f"Loading data from compressed NumPy archive: '{npz_path}'...")
        with np.load(npz_path, allow_pickle=True) as data:
            X = data["X"].astype(np.float32)
            y = data["y"].astype(np.int64)
            mapping = data["mapping"].tolist()
        print(f"Loaded {X.shape[0]} samples in shape {X.shape[1:]} (time_frames, n_mfcc).")
        return X, y, mapping

    elif os.path.exists(json_path):
        print(f"Loading data from JSON file: '{json_path}'...")
        with open(json_path, "r") as fp:
            data = json.load(fp)
        X = np.array(data["mfcc"], dtype=np.float32)
        y = np.array(data["labels"], dtype=np.int64)
        mapping = data["mapping"]
        print(f"Loaded {X.shape[0]} samples.")
        return X, y, mapping

    else:
        raise FileNotFoundError(
            f"No processed dataset found! Expected '{npz_path}' or '{json_path}'.\n"
            "Please run 'python dataset_prep.py' first!"
        )


def build_cnn_model(input_shape, num_classes=10):
    """
    Builds a 2D CNN using the Keras Functional API.
    Guarantees 100% compatibility across Keras 2.x and Keras 3.x / TF 2.16+.
    """
    inputs = layers.Input(shape=input_shape, name="audio_mfcc_input")

    # Convolutional Block 1
    x = layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
    x = layers.Dropout(0.25)(x)

    # Convolutional Block 2
    x = layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
    x = layers.Dropout(0.25)(x)

    # Convolutional Block 3
    x = layers.Conv2D(128, kernel_size=(2, 2), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)
    x = layers.Dropout(0.30)(x)

    # Classification Head
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.40)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="genre_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Music_Genre_Classifier_CNN")
    return model


def plot_training_history(history, output_dir=PLOTS_DIR):
    """Plots and saves loss and accuracy curves."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    axs[0].plot(history.history["accuracy"], label="Train Accuracy", color="#2563eb", lw=2)
    axs[0].plot(history.history["val_accuracy"], label="Validation Accuracy", color="#ea580c", lw=2)
    axs[0].set_title("Training vs Validation Accuracy", fontsize=13, fontweight="bold")
    axs[0].set_xlabel("Epoch", fontsize=11)
    axs[0].set_ylabel("Accuracy", fontsize=11)
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Loss Plot
    axs[1].plot(history.history["loss"], label="Train Loss", color="#2563eb", lw=2)
    axs[1].plot(history.history["val_loss"], label="Validation Loss", color="#ea580c", lw=2)
    axs[1].set_title("Training vs Validation Loss", fontsize=13, fontweight="bold")
    axs[1].set_xlabel("Epoch", fontsize=11)
    axs[1].set_ylabel("Loss", fontsize=11)
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved training history curves to '{chart_path}'")


def plot_confusion_matrix(y_true, y_pred, mapping, output_dir=PLOTS_DIR):
    """Plots and saves a formatted confusion matrix."""
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=mapping, yticklabels=mapping
    )
    plt.title("GTZAN Genre Classification - Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Genre", fontsize=12)
    plt.ylabel("True Genre", fontsize=12)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix heatmap to '{cm_path}'")


def train():
    # 1. Load Data
    X, y, mapping = load_dataset()

    # 2. 80/20 Stratified Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Reshape input to 4D for 2D Conv: (samples, time_frames, n_mfcc, 1)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]

    print("\n" + "=" * 65)
    print(f"Dataset Split Summary:")
    print(f"  Training samples  : {X_train.shape[0]} (80%)")
    print(f"  Validation samples: {X_val.shape[0]} (20%)")
    print(f"  CNN Input shape   : {X_train.shape[1:]}")
    print(f"  Genres ({len(mapping)})       : {mapping}")
    print("=" * 65 + "\n")

    # 3. Build Model
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_cnn_model(input_shape, num_classes=len(mapping))

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # 4. Callbacks
    os.makedirs(MODEL_DIR, exist_ok=True)
    training_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    # 5. Model Training
    print("\nStarting CNN Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=training_callbacks,
        verbose=1
    )

    # 6. Evaluation
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print("\n" + "=" * 65)
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Validation Loss    : {val_loss:.4f}")
    print("=" * 65)

    # 7. Classification Report & Confusion Matrix
    y_pred_probs = model.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=mapping))

    plot_training_history(history)
    plot_confusion_matrix(y_val, y_pred, mapping)

    # 8. Save Model and Genre Mapping
    model.save(MODEL_SAVE_PATH)
    with open(MAPPING_SAVE_PATH, "w") as fp:
        json.dump(mapping, fp, indent=2)

    print(f"\nModel saved successfully to: '{MODEL_SAVE_PATH}'")
    print(f"Genre mapping saved to     : '{MAPPING_SAVE_PATH}'")


if __name__ == "__main__":
    train()
