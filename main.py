import os.path

import keras
import numpy as np
from keras import Sequential, Input
from keras.models import Model
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.src.legacy import layers
from keras.src.optimizers import Adam
import tensorflow as tf
from tensorflow import data as tf_data
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

from augment_data import augment_train_data
from class_to_number import class_names
from constants import IMG_SIZE, IMG_CHANNEL_NUMBER, EPOCHS, BATCH_SIZE, INPUT_SHAPE, NUM_CLASSES
from describe_data import describe_data
from load_dataset import load_dataset_from_images
import matplotlib.pyplot as plt
import wandb

from model_builder import build_model
from utils import create_run_name, create_checkpoint_name, create_model_name
from visualizing_utils import plot_metric

train_ds, val_ds = load_dataset_from_images()

train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

describe_data(train_ds, val_ds)

# MODEL
# basic = Sequential(
#     layers=[
#         Conv2D(
#             filters=32, #
#             kernel_size=(5, 5), #
#             input_shape=(IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNEL_NUMBER), #
#             activation='relu' #
#         ),
#         BatchNormalization(),
#         MaxPooling2D(
#             pool_size=(2, 2)
#         ),
#         Conv2D(64, (3, 3), activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D(pool_size=(2, 2)),
#         Conv2D(128, (3, 3), activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D(pool_size=(2, 2)),
#         Flatten(),
#         Dense(256, activation='relu'),
#         Dense(len(class_names), activation='softmax')
#     ],
#     trainable=True,
#     name=None
# )
# basic.compile(
#     optimizer=Adam(learning_rate=0.1),
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )
# basic.summary()

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
    dropout=0.5,  # można ustawić inną wartość niż dropout_f
    dropout_f=0.5,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
)

wandb.init(project="polish-road-signs-classification", name=create_run_name(config=hyperparams), config=hyperparams)
wandb_config = wandb.config

model = build_model(wandb_config, INPUT_SHAPE, NUM_CLASSES)

# na ten moment wsparcie tylko dla optimizer Adam
if wandb.config['optimizer'] == 'adam':
    optimizer = Adam(learning_rate=wandb.config['learning_rate'])
else:
    raise ValueError(f"Unknown optimizer: {wandb.config['optimizer']}")


model.compile(optimizer=optimizer,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

# Print model summary
model.summary()

history = model.fit( # w przykładach jest rozbicie na x i y, wywala error ConnectionAbortedError: [WinError 10053] Nawiązane połączenie zostało przerwane przez oprogramowanie zainstalowane w komputerze-hoście
    train_ds,
    steps_per_epoch=int(np.ceil(len(list(train_ds)) / 10)),
    epochs=wandb.config['epochs'],
    batch_size=wandb.config['batch_size'],
    validation_data=val_ds,
    validation_steps=int(np.ceil(len(list(val_ds)) / 10)),
    verbose=1,
    callbacks=[  #  rozwazyc dodanie EarlyStopping
        WandbMetricsLogger(log_freq=1),
        WandbModelCheckpoint(filepath=os.path.join("checkpoints", create_checkpoint_name(), "checkpoint_{epoch:02d}.keras"),
                             save_freq="epoch")
    ]
)

model.save(f'best_models/{create_model_name()}.keras')

wandb.finish(exit_code=0)

# final = Sequential([
#     Conv2D(60, (5, 5), input_shape=(IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNEL_NUMBER), activation='relu'),
#     Conv2D(60, (5, 5), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#
#     Conv2D(30, (3, 3), activation='relu'),
#     Conv2D(30, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#
#     Flatten(),
#     Dense(500, activation='relu'),
#     Dropout(0.5),
#     Dense(len(class_names), activation='softmax')
# ])
# final.compile(Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# final.summary()
#
# history = final.fit(
#     train_ds,
#     steps_per_epoch=int(np.ceil(len(list(train_ds)) / 10)),
#     epochs=EPOCHS,
#     validation_data=val_ds,
#     validation_steps=int(np.ceil(len(list(val_ds)) / 10)),
#     verbose=1
# )

# plot_metric(history=history, metric_name='accuracy')
# plot_metric(history=history, metric_name='loss')
