import os

import keras
from keras import Sequential
from keras.src.applications.vgg16 import preprocess_input
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.optimizers import Adam
import numpy as np
from keras.src.saving import load_model
from keras.src.utils import load_img, img_to_array

from class_to_number import class_names
from constants import IMG_SIZE, IMG_CHANNEL_NUMBER, SAVED_MODEL_IDS, SAVED_MODEL_DIR, EPOCHS, BATCH_SIZE


def create_model():
    model = Sequential([
        Conv2D(60, (5, 5), input_shape=(IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNEL_NUMBER), activation='relu'),
        Conv2D(60, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(30, (3, 3), activation='relu'),
        Conv2D(30, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(500, activation='relu'),
        Dropout(0.5),
        Dense(len(class_names), activation='softmax')
    ])
    model.compile(Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

def train_model(model, train_ds, val_ds):
    history = model.fit(
        x=train_ds,
        y=None,  # Targets are provided directly by the dataset
        batch_size=None,  # Use default batch size for datasets
        epochs=EPOCHS,  # Predefined constant for the number of epochs
        verbose=1,  # Show progress bar during training
        callbacks=None,  # No additional callbacks specified
        validation_split=0.0,  # No splitting as validation data is separate
        validation_data=val_ds,  # Predefined validation dataset
        shuffle=True,  # Shuffle training data at the beginning of each epoch
        class_weight=None,  # No class weighting applied
        sample_weight=None,  # No sample weighting applied
        initial_epoch=0,  # Start from the first epoch
        steps_per_epoch=None,  # Calculate training steps
        validation_steps=None,  # Calculate validation steps
        validation_batch_size=None,  # Use the default batch size for validation
        validation_freq=1,  # Perform validation after every epoch
    )

    return history

def save_model(model):
    if not os.path.exists(SAVED_MODEL_IDS):
        current_id = 0
    else:
        with open(SAVED_MODEL_IDS, "r") as f:
            current_id = int(f.read().strip())

    new_id = current_id + 1

    with open(SAVED_MODEL_IDS, "w") as f:
        f.write(str(new_id))

    model_save_path = os.path.join(SAVED_MODEL_DIR, f"model_{new_id}.keras")

    try:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # Ensure the directory exists
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")
    except PermissionError:
        print(
            f"Permission denied when trying to save the model to {model_save_path}. Please check your write permissions.")
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")

def load_model_from_file(model_id):
    model_save_path = os.path.join(SAVED_MODEL_DIR, str(model_id))

    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"No saved model found at {model_save_path}")

    model = load_model(model_save_path)
    print(f"Model loaded from: {model_save_path}")

    return model

def load_latest_model():
    if not os.path.exists(SAVED_MODEL_IDS):
        raise FileNotFoundError("No saved model IDs found.")

    with open(SAVED_MODEL_IDS, "r") as f:
        latest_id = int(f.read().strip())

    return load_model_from_file(latest_id)

def predict_image(model, img_path):
    img = load_img(img_path, target_size=(128, 128))

    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)

    return prediction