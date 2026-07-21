"""
Image Classification Model
============================
CNN-based image classifier with support for:
- Training from scratch (custom CNN)
- Transfer learning (MobileNetV2 / ResNet50)
- Saving/loading trained models
- Evaluation with metrics and confusion matrix
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt


class ImageClassifier:
    def __init__(self, img_size=(224, 224), model_dir="saved_model"):
        self.img_size = img_size
        self.model_dir = model_dir
        self.model = None
        self.class_names = []
        os.makedirs(model_dir, exist_ok=True)

    # ─────────────────────────────────────────────
    # Data Loading
    # ─────────────────────────────────────────────

    def load_data(self, dataset_path="dataset", batch_size=32, val_split=0.2, augment=True):
        """
        Load images from directory structure:
            dataset/
                class1/
                    img1.jpg
                class2/
                    img1.jpg
        Returns train and validation generators.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path '{dataset_path}' not found.")

        if augment:
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255,
                rotation_range=20,
                width_shift_range=0.15,
                height_shift_range=0.15,
                shear_range=0.1,
                zoom_range=0.15,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],
                validation_split=val_split
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=val_split)

        val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=val_split)

        train_gen = train_datagen.flow_from_directory(
            dataset_path, target_size=self.img_size, batch_size=batch_size,
            class_mode="categorical", subset="training", shuffle=True
        )

        val_gen = val_datagen.flow_from_directory(
            dataset_path, target_size=self.img_size, batch_size=batch_size,
            class_mode="categorical", subset="validation", shuffle=False
        )

        self.class_names = list(train_gen.class_indices.keys())
        print(f"[INFO] Found {len(self.class_names)} classes: {self.class_names}")
        print(f"[INFO] Training samples: {train_gen.samples} | Validation samples: {val_gen.samples}")

        return train_gen, val_gen

    # ─────────────────────────────────────────────
    # Model Architectures
    # ─────────────────────────────────────────────

    def build_custom_cnn(self, num_classes):
        """Build a CNN from scratch — good for learning, small-to-medium datasets."""
        model = models.Sequential([
            layers.Input(shape=(*self.img_size, 3)),

            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),

            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        self.model = model
        print("[INFO] Custom CNN model built.")
        model.summary()
        return model

    def build_transfer_model(self, num_classes, base="mobilenet", fine_tune=False):
        """
        Build a transfer-learning model using a pretrained backbone.
        base: 'mobilenet' (fast, lightweight) or 'resnet50' (more accurate, heavier)
        """
        input_shape = (*self.img_size, 3)

        if base == "mobilenet":
            base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
        elif base == "resnet50":
            base_model = ResNet50(input_shape=input_shape, include_top=False, weights="imagenet")
        else:
            raise ValueError("base must be 'mobilenet' or 'resnet50'")

        base_model.trainable = fine_tune

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax")
        ])

        lr = 1e-5 if fine_tune else 1e-3
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        self.model = model
        print(f"[INFO] Transfer learning model built (base={base}, fine_tune={fine_tune}).")
        model.summary()
        return model

    # ─────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────

    def train(self, train_gen, val_gen, epochs=20, patience=5):
        """Train the model with early stopping and checkpointing."""
        if self.model is None:
            raise RuntimeError("Build a model first with build_custom_cnn() or build_transfer_model().")

        checkpoint_path = os.path.join(self.model_dir, "best_model.keras")

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
            ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1)
        ]

        print(f"[INFO] Starting training for {epochs} epochs...")
        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks
        )

        self.save_model()
        self.plot_history(history)
        return history

    def plot_history(self, history):
        """Plot and save training accuracy/loss curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(history.history["accuracy"], label="Train")
        axes[0].plot(history.history["val_accuracy"], label="Validation")
        axes[0].set_title("Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(history.history["loss"], label="Train")
        axes[1].plot(history.history["val_loss"], label="Validation")
        axes[1].set_title("Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.model_dir, "training_history.png")
        plt.savefig(plot_path, dpi=120)
        print(f"[INFO] Training history plot saved to {plot_path}")
        plt.close()

    # ─────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────

    def evaluate(self, val_gen):
        """Evaluate model and print classification report + confusion matrix."""
        from sklearn.metrics import classification_report, confusion_matrix
        import seaborn as sns

        loss, acc = self.model.evaluate(val_gen, verbose=0)
        print(f"\n[RESULT] Validation Loss: {loss:.4f} | Validation Accuracy: {acc:.4f}\n")

        val_gen.reset()
        y_true = val_gen.classes
        y_pred_probs = self.model.predict(val_gen, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        print("[CLASSIFICATION REPORT]")
        print(classification_report(y_true, y_pred, target_names=self.class_names))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        cm_path = os.path.join(self.model_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=120)
        print(f"[INFO] Confusion matrix saved to {cm_path}")
        plt.close()

        return loss, acc

    # ─────────────────────────────────────────────
    # Save / Load
    # ─────────────────────────────────────────────

    def save_model(self, name="final_model.keras"):
        """Save model weights and class names."""
        model_path = os.path.join(self.model_dir, name)
        self.model.save(model_path)

        meta_path = os.path.join(self.model_dir, "class_names.json")
        with open(meta_path, "w") as f:
            json.dump({"class_names": self.class_names, "img_size": self.img_size}, f, indent=2)

        print(f"[INFO] Model saved to {model_path}")
        print(f"[INFO] Class names saved to {meta_path}")

    def load_model(self, name="final_model.keras"):
        """Load a previously trained model and its class names."""
        model_path = os.path.join(self.model_dir, name)
        meta_path = os.path.join(self.model_dir, "class_names.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = tf.keras.models.load_model(model_path)

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self.class_names = meta["class_names"]
            self.img_size = tuple(meta["img_size"])

        print(f"[INFO] Model loaded from {model_path}")
        print(f"[INFO] Classes: {self.class_names}")

    # ─────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────

    def predict_image(self, image_path, top_k=3):
        """Predict the class of a single image. Returns sorted (class, confidence) list."""
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        img = tf.keras.utils.load_img(image_path, target_size=self.img_size)
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = self.model.predict(img_array, verbose=0)[0]
        top_indices = np.argsort(preds)[::-1][:top_k]

        results = [(self.class_names[i], float(preds[i]) * 100) for i in top_indices]
        return results

    def predict_batch(self, folder_path, top_k=1):
        """Predict classes for all images in a folder."""
        results = {}
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp")

        for fname in os.listdir(folder_path):
            if fname.lower().endswith(valid_ext):
                fpath = os.path.join(folder_path, fname)
                preds = self.predict_image(fpath, top_k=top_k)
                results[fname] = preds

        return results
