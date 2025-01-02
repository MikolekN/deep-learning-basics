# DATA AUGMENTATION

import matplotlib.pyplot as plt
from keras import layers
from tensorflow import data as tf_data

from constants import DEBUG

# TODO: chyba jednak nierobić, bo może się zmienić znaczenie znaku
def augment_train_data(train_ds):
    augmentation_pipeline = layers.Pipeline([
        layers.RandomFlip(
            mode="horizontal_and_vertical",
            seed=184474
        ),
        layers.RandomRotation(
            factor=0.1,
            fill_mode="constant",
            interpolation="bilinear",
            seed=184474,
            fill_value=0.0,
            data_format="channels_last"
        ),
    ])

    train_ds = train_ds.map(
        lambda x, y: (augmentation_pipeline(x, training=True), y),
        num_parallel_calls=tf_data.AUTOTUNE,
    )

    if DEBUG:
        for images, _ in train_ds.take(1):
            augmented_images = augmentation_pipeline(images)
            plt.figure(figsize=(10, 10))
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(augmented_images[i].numpy().astype("uint8"))
                plt.axis("off")
            plt.show()

    return train_ds