# GENERATE DATASET FROM IMAGES

import keras
import matplotlib.pyplot as plt
from tensorflow import data as tf_data
from class_to_number import class_names
from constants import TRAINING_DATA_DIR, IMG_BATCH_SIZE, IMG_SIZE, DEBUG
from describe_data import describe_data


def _load_dataset_from_images():
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        directory=TRAINING_DATA_DIR,
        labels="inferred", # labels are generated from the directory structure
        label_mode="int", # labels are encoded as integers
        class_names=class_names,
        color_mode="rgb",
        batch_size=IMG_BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=184474, # shuffle and split are the same if set to the same value (used for debugging, testing, or reproducible experiments)
        validation_split=0.2,
        subset="both",
        interpolation="bilinear", # determines the method used to resize images when their dimensions do not match the specified image_size
        follow_links=False, # whether to visit subdirectories pointed to by symlinks
        crop_to_aspect_ratio=False, # images will not be cropped
        pad_to_aspect_ratio=True, # images will be padded to keep aspect ratio
        data_format=None,
        verbose=DEBUG # display number information on classes and number of files found
    )

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

    return train_ds, val_ds


def load_dataset():
    train_ds, val_ds = _load_dataset_from_images()

    train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

    describe_data(train_ds, val_ds)

    return train_ds, val_ds
