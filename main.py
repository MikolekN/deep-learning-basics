import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
import tensorflow as tf

from class_to_number import class_names
from constants import TRAINING_DATA_DIR, IMG_BATCH_SIZE, IMG_SIZE, DEBUG

# GENERATE DATASET FROM IMAGES
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
    verbose=True # display number information on classes and number of files found
)

if DEBUG:
    for images, labels in train_ds.take(1):
        print(f"Image batch shape: {images.shape} (Batch of {images.shape[0]} images, each resized to {images.shape[1]}x{images.shape[2]} pixels with {images.shape[3]} color channels)")
        print(f"Label batch shape: {labels.shape} (Batch of {labels.shape[0]} labels, one for each image)")

        plt.figure(figsize=(10, 10))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i].numpy()])
            plt.axis("off")
        plt.show()


# DATA AUGMENTATION
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

train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

def describe_dataset(dataset_name, element_spec):
    print(f"{dataset_name} structure:")
    for i, element in enumerate(element_spec):
        if isinstance(element, tf.TensorSpec):
            shape = element.shape
            dtype = element.dtype
            print(f"  Element {i + 1}:")
            print(f"    - Shape: {shape}  - The dimensions of the tensor. 'None' indicates dynamic batch size.")

            if len(shape) == 4 and shape[-1] == 3:  # Likely an image tensor
                print(f"    - Data type: {dtype} - A batch of RGB images.")
            elif len(shape) == 1:  # Likely a label tensor
                print(f"    - Data type: {dtype} - A batch of integer labels.")
        else:
            print(f"  Element {i + 1}: Unknown type or structure")

if DEBUG:
    describe_dataset("Train dataset", train_ds.element_spec)
    describe_dataset("Validation dataset", val_ds.element_spec)

# MODEL
# def make_model(input_shape, num_classes):
#     inputs = keras.Input(shape=input_shape)
#
#     # Entry block
#     x = layers.Rescaling(1.0 / 255)(inputs)
#     x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)
#
#     previous_block_activation = x  # Set aside residual
#
#     for size in [256, 512, 728]:
#         x = layers.Activation("relu")(x)
#         x = layers.SeparableConv2D(size, 3, padding="same")(x)
#         x = layers.BatchNormalization()(x)
#
#         x = layers.Activation("relu")(x)
#         x = layers.SeparableConv2D(size, 3, padding="same")(x)
#         x = layers.BatchNormalization()(x)
#
#         x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
#
#         # Project residual
#         residual = layers.Conv2D(size, 1, strides=2, padding="same")(
#             previous_block_activation
#         )
#         x = layers.add([x, residual])  # Add back residual
#         previous_block_activation = x  # Set aside next residual
#
#     x = layers.SeparableConv2D(1024, 3, padding="same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)
#
#     x = layers.GlobalAveragePooling2D()(x)
#     if num_classes == 2:
#         units = 1
#     else:
#         units = num_classes
#
#     x = layers.Dropout(0.25)(x)
#     # We specify activation=None so as to return logits
#     outputs = layers.Dense(units, activation=None)(x)
#     return keras.Model(inputs, outputs)
#
#
# model = make_model(input_shape=image_size + (3,), num_classes=2)
# keras.utils.plot_model(model, show_shapes=True)
