import os
from datetime import datetime

import tensorflow as tf
from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.optimizers import Adam, SGD
from keras.src.saving import load_model
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

import wandb
from constants import DEBUG, SAVED_MODEL_DIR
from utils import create_checkpoint_name


class SparsePrecision(tf.keras.metrics.Metric):
    def __init__(self, name="sparse_precision", **kwargs):
        super(SparsePrecision, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=1)
        y_true = tf.cast(y_true, dtype=tf.int32)
        y_pred = tf.cast(y_pred, dtype=tf.int32)

        tp = tf.reduce_sum(tf.cast(tf.logical_and(y_pred == 1, y_true == 1), dtype=tf.float32))
        fp = tf.reduce_sum(tf.cast(tf.logical_and(y_pred == 1, y_true == 0), dtype=tf.float32))

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        return precision

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)


class SparseRecall(tf.keras.metrics.Metric):
    def __init__(self, name="sparse_recall", **kwargs):
        super(SparseRecall, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=1)
        y_true = tf.cast(y_true, dtype=tf.int32)
        y_pred = tf.cast(y_pred, dtype=tf.int32)

        tp = tf.reduce_sum(tf.cast(tf.logical_and(y_pred == 1, y_true == 1), dtype=tf.float32))
        fn = tf.reduce_sum(tf.cast(tf.logical_and(y_pred == 0, y_true == 1), dtype=tf.float32))

        self.true_positives.assign_add(tp)
        self.false_negatives.assign_add(fn)

    def result(self):
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        return recall

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_negatives.assign(0)


class SparseF1Score(tf.keras.metrics.Metric):
    def __init__(self, name="sparse_f1_score", **kwargs):
        super(SparseF1Score, self).__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name="f1", initializer="zeros")
        self.precision_metric = SparsePrecision()
        self.recall_metric = SparseRecall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision_metric.update_state(y_true, y_pred)
        self.recall_metric.update_state(y_true, y_pred)

        precision = self.precision_metric.result()
        recall = self.recall_metric.result()

        f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
        self.f1.assign(f1)

    def result(self):
        return self.f1

    def reset_states(self):
        self.f1.assign(0)
        self.precision_metric.reset_states()
        self.recall_metric.reset_states()


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
    elif wandb.config['optimizer'] == 'sgd':
        optimizer = SGD(learning_rate=wandb.config['learning_rate'])
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
        ]
    )
    model.summary()

    return model


def train_model(model, train_ds, val_ds):
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)

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
                                 save_freq="epoch"),
            callback
        ]
    )

    val_loss, val_acc, val_precision, val_recall, val_f1 = model.evaluate(val_ds, verbose=DEBUG)
    wandb.log({
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1
    })

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
