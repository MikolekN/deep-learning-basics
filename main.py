import keras
import numpy as np
from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.src.legacy import layers
from keras.src.optimizers import Adam
from tensorflow import data as tf_data

from augment_data import augment_train_data
from class_to_number import class_names
from constants import IMG_SIZE, IMG_CHANNEL_NUMBER
from describe_data import describe_data
from load_dataset import load_dataset_from_images
import matplotlib.pyplot as plt
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

final = Sequential([
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
final.compile(Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
final.summary()

EPOCHS = 3
BATCH_SIZE = 32

history = final.fit(
    train_ds,
    steps_per_epoch=int(np.ceil(len(list(train_ds)) / 10)),
    epochs=10,
    validation_data=val_ds,
    validation_steps=int(np.ceil(len(list(val_ds)) / 10)),
    verbose=1
)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
plt.show()
