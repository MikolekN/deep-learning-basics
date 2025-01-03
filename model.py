import os

import numpy as np
from keras import Input
from keras.models import Model
from keras.src.applications.vgg16 import preprocess_input
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.optimizers import Adam
from keras.src.saving import load_model
from keras.src.utils import load_img, img_to_array
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

import wandb
from constants import SAVED_MODEL_IDS, SAVED_MODEL_DIR, EPOCHS, BATCH_SIZE
from utils import create_run_name, create_checkpoint_name, create_model_name

config = {
    'filters': 60,                    # Number of filters for the first layer
    'kernel_1': (5, 5),               # Kernel size for the first set of Conv2D layers
    'kernel_2': (3, 3),               # Kernel size for the second set of Conv2D layers
    'padding': 'valid',               # Padding type
    'activation': 'relu',             # Activation function
    'pooling': (2, 2),                # Pooling size for MaxPooling2D
    'dropout': 0.5,                   # Dropout rate for intermediate layers
    'dropout_f': 0.5,                 # Dropout rate for fully connected layers
    'dense_units': 500,               # Number of units in the dense layer
    'learning_rate': 0.001            # Learning rate for the Adam optimizer
}

hyperparams = dict(
    filters=60,
    kernel_1=(5, 5),
    kernel_2=(3, 3),
    padding='valid',
    pooling=(2, 2),
    learning_rate=0.001,
    wd=0.0,
    learning_rate_schedule='RLR',    # cos, cyclic, step decay
    optimizer='adam',     # RMS
    dense_units=500,
    activation='relu',      # elu, LeakyRelu
    dropout=0.5,  # can be different value than dropout_f, place for experiments
    dropout_f=0.5,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
)


def _initialize():
    wandb.init(project="polish-road-signs-classification", name=create_run_name(config=hyperparams), config=hyperparams)
    return wandb.config


def _uninitialize():
    wandb.finish(exit_code=0)


# def create_model():
#     model = Sequential([
#         Conv2D(60, (5, 5), input_shape=(IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNEL_NUMBER), activation='relu'),
#         Conv2D(60, (5, 5), activation='relu'),
#         MaxPooling2D(pool_size=(2, 2)),
#
#         Conv2D(30, (3, 3), activation='relu'),
#         Conv2D(30, (3, 3), activation='relu'),
#         MaxPooling2D(pool_size=(2, 2)),
#
#         Flatten(),
#         Dense(500, activation='relu'),
#         Dropout(0.5),
#         Dense(len(class_names), activation='softmax')
#     ])
#     model.compile(Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     model.summary()
#
#     return model


def create_model(config, input_shape, num_classes):
    inp = Input(shape=input_shape)

    # Layer-1: Conv2D-Conv2D-Pooling
    x = Conv2D(filters=config['filters'], kernel_size=config['kernel_1'], activation=config['activation'], padding=config['padding'])(inp)
    x = Conv2D(filters=config['filters'], kernel_size=config['kernel_1'], activation=config['activation'], padding=config['padding'])(x)
    x = MaxPooling2D(pool_size=config['pooling'])(x)

    # Layer-2: Conv2D-Conv2D-Pooling
    x = Conv2D(filters=config['filters'] // 2, kernel_size=config['kernel_2'], activation=config['activation'], padding=config['padding'])(x)
    x = Conv2D(filters=config['filters'] // 2, kernel_size=config['kernel_2'], activation=config['activation'], padding=config['padding'])(x)
    x = MaxPooling2D(pool_size=config['pooling'])(x)

    # Flatten and Fully Connected Head
    x = Flatten()(x)
    x = Dense(config['dense_units'], activation=config['activation'])(x)
    x = Dropout(config['dropout_f'])(x)

    out = Dense(num_classes, activation='softmax')(x)

    # Build and compile the model
    model = Model(inputs=inp, outputs=out)

    # na ten moment wsparcie tylko dla optimizer Adam
    if config['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=config['learning_rate'])
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    return model


def train_model(model, train_ds, val_ds):
    history = model.fit(
        x=train_ds,
        y=None,  # Targets are provided directly by the dataset
        epochs=wandb.config['epochs'],
        batch_size=wandb.config['batch_size'],
        verbose=1,  # Show progress bar during training
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


def save_model(model):
    model_save_path = os.path.join(SAVED_MODEL_DIR, f"model_{create_model_name()}.keras")

    try:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # Ensure the directory exists
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")
    except PermissionError:
        print(
            f"Permission denied when trying to save the model to {model_save_path}. Please check your write permissions.")
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")


def load_model_from_file(model_file):
    model_path = os.path.join(SAVED_MODEL_DIR, model_file)

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
