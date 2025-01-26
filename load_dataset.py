import keras
import matplotlib.pyplot as plt
from tensorflow import data as tf_data
from class_to_number import class_names
from constants import (
    TRAINING_DATA_DIR,
    TRAINING_DATA_SUPPLEMENT_DIR,
    IMG_BATCH_SIZE,
    IMG_SIZE, DEBUG,
    TEST_DATA_DIR
)
from describe_data import describe_data

def _load_dataset_supplemented():
    primary_ds = keras.utils.image_dataset_from_directory(
        directory=TRAINING_DATA_DIR,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        color_mode="rgb",
        batch_size=IMG_BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=184474,
        pad_to_aspect_ratio=True,
        data_format=None,
        verbose=DEBUG
    )

    supplement_ds = keras.utils.image_dataset_from_directory(
        directory=TRAINING_DATA_SUPPLEMENT_DIR,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        color_mode="rgb",
        batch_size=IMG_BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=184474,
        pad_to_aspect_ratio=True,
        data_format=None,
        verbose=DEBUG
    )

    combined_ds = primary_ds.concatenate(supplement_ds)
    val_ds = combined_ds.take(int(len(combined_ds) * 0.2))
    train_ds = combined_ds.skip(int(len(combined_ds) * 0.2))

    return train_ds, val_ds

def _load_dataset():
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        directory=TRAINING_DATA_DIR,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        color_mode="rgb",
        batch_size=IMG_BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=184474,
        validation_split=0.2,
        subset="both",
        pad_to_aspect_ratio=True,
        data_format=None,
        verbose=DEBUG
    )

    return train_ds, val_ds

def load_training_dataset(supplemented=False):
    if supplemented:
        train_ds, val_ds = _load_dataset_supplemented()
    else:
        train_ds, val_ds = _load_dataset()

    if DEBUG:
        for images, labels in train_ds.take(1):
            print(f"Image batch shape: {images.shape} (Batch of {images.shape[0]} images, each resized to {images.shape[1]}x{images.shape[2]} pixels with {images.shape[3]} color channels)")
            print(f"Label batch shape: {labels.shape} (Batch of {labels.shape[0]} labels, one for each image)")

            plt.figure(figsize=(10, 10))
            for i in range(9):
                plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i].numpy()])
                plt.axis("off")
            plt.show()

    train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

    describe_data(train_ds, val_ds)

    return train_ds, val_ds

def load_testing_dataset(source=TEST_DATA_DIR):
    return keras.utils.image_dataset_from_directory(
        directory=source,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        color_mode="rgb",
        batch_size=IMG_BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=False
    )