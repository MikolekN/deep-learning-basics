import os
from datetime import datetime

import keras
import numpy as np
from keras import Sequential
from keras.src.applications.vgg16 import preprocess_input
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.optimizers import Adam
from keras.src.saving import load_model
from keras.src.utils import load_img, img_to_array
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

import wandb
from constants import DEBUG, SAVED_MODEL_DIR
from utils import create_checkpoint_name
import tensorflow as tf


class SparsePrecision(tf.keras.metrics.Metric):
    def __init__(self, name="sparse_precision", **kwargs):
        super(SparsePrecision, self).__init__(name=name, **kwargs)
        self.precision = self.add_weight(name="precision", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=1)  # Convert to class indices
        y_true = tf.cast(y_true, dtype=tf.int32)  # Ensure true labels are in int32 format
        y_pred = tf.cast(y_pred, dtype=tf.int32)  # Ensure predicted labels are in int32 format

        # True positives
        true_positives = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), dtype=tf.float32))

        # Predicted positives
        predicted_positives = tf.reduce_sum(tf.cast(y_pred, dtype=tf.float32))

        # Precision
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        self.precision.assign(precision)

    def result(self):
        return self.precision

    def reset_states(self):
        self.precision.assign(0)


class SparseRecall(tf.keras.metrics.Metric):
    def __init__(self, name="sparse_recall", **kwargs):
        super(SparseRecall, self).__init__(name=name, **kwargs)
        self.recall = self.add_weight(name="recall", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=1)  # Convert to class indices
        y_true = tf.cast(y_true, dtype=tf.int32)  # Ensure true labels are in int32 format
        y_pred = tf.cast(y_pred, dtype=tf.int32)  # Ensure predicted labels are in int32 format

        # True positives
        true_positives = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), dtype=tf.float32))

        # Possible positives
        possible_positives = tf.reduce_sum(tf.cast(y_true, dtype=tf.float32))

        # Recall
        recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
        self.recall.assign(recall)

    def result(self):
        return self.recall

    def reset_states(self):
        self.recall.assign(0)


class SparseF1Score(tf.keras.metrics.Metric):
    def __init__(self, name="sparse_f1_score", **kwargs):
        super(SparseF1Score, self).__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name="f1", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=1)  # Convert to class indices
        y_true = tf.cast(y_true, dtype=tf.int32)  # Ensure true labels are in int32 format
        y_pred = tf.cast(y_pred, dtype=tf.int32)  # Ensure predicted labels are in int32 format

        # True positives
        true_positives = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), dtype=tf.float32))

        # Predicted positives
        predicted_positives = tf.reduce_sum(tf.cast(y_pred, dtype=tf.float32))

        # Possible positives
        possible_positives = tf.reduce_sum(tf.cast(y_true, dtype=tf.float32))

        # Precision and recall
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        recall = true_positives / (possible_positives + tf.keras.backend.epsilon())

        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        self.f1.assign(f1)

    def result(self):
        return self.f1

    def reset_states(self):
        self.f1.assign(0)

def create_model(input_shape, num_classes, config=None):
    config = config or wandb.config

    model = Sequential([
        Conv2D(config['filters'], config['kernel_1'], activation=config['activation'], padding=config['padding'], input_shape=input_shape),
        Conv2D(config['filters'], config['kernel_1'], activation=config['activation'], padding=config['padding']),
        MaxPooling2D(pool_size=config['pooling']),

        Conv2D(config['filters'] // 2, config['kernel_2'], activation=config['activation'], padding=config['padding']),
        Conv2D(config['filters'] // 2, config['kernel_2'], activation=config['activation'], padding=config['padding']),
        MaxPooling2D(pool_size=config['pooling']),

        Flatten(),
        Dense(config['dense_units'], activation=config['activation']),
        Dropout(config['dropout_f']),
        Dense(num_classes, activation='softmax')
    ])

    if config['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=config['learning_rate'])
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            SparsePrecision(name='precision'),
            SparseRecall(name='recall'),
            SparseF1Score(name='f1_score')
        ])
    model.summary()

    return model


def train_model(model, train_ds, val_ds):
    history = model.fit(
        x=train_ds,
        y=None,  # Targets are provided directly by the dataset
        epochs=wandb.config['epochs'],
        batch_size=wandb.config['batch_size'],
        verbose=DEBUG,  # Show progress bar during training
        validation_split=0.0,  # No splitting as validation data is separate
        validation_data=val_ds,  # Predefined validation dataset
        shuffle=True,  # Shuffle training data at the beginning of each epoch
        class_weight=None,  # No class weighting applied
        sample_weight=None,  # No sample weighting applied
        initial_epoch=0,  # Start from the first epoch
        steps_per_epoch=None,  # Calculate training steps
        #steps_per_epoch=int(np.ceil(len(list(train_ds)) / 10)),
        validation_steps=None,  # Calculate validation steps
        #validation_steps=int(np.ceil(len(list(val_ds)) / 10)),
        validation_batch_size=None,  # Use the default batch size for validation
        validation_freq=1,  # Perform validation after every epoch
        callbacks=[  # Add EarlyStopping in next project phase
            WandbMetricsLogger(log_freq=1),
            WandbModelCheckpoint(filepath=os.path.join("checkpoints", create_checkpoint_name(), "checkpoint_{epoch:02d}.keras"),
                                 save_freq="epoch")
        ]
    )

    return history


def create_model_name() -> str:
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"model_{current_datetime}.keras"


def assemble_model_name(timestamp: datetime) -> str:
    return f"model_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.keras"


def save_model(model):
    model_save_path = os.path.join(SAVED_MODEL_DIR, create_model_name())

    try:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")
    except PermissionError:
        print(f"Permission denied when trying to save the model to {model_save_path}.")
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")


def load_model_from_name(model_name: str):
    return _load_model(model_name)


def load_model_from_timestamp(timestamp: datetime):
    model_name = assemble_model_name(timestamp)
    return _load_model(model_name)


def _load_model(model_file_name: str):
    model_path = os.path.join(SAVED_MODEL_DIR, model_file_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model found at {model_path}")

    model = load_model(model_path)
    print(f"Model loaded from: {model_path}")

    return model


def predict_image(model, img_path):
    img = load_img(img_path, target_size=(128, 128))

    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)

    return prediction
