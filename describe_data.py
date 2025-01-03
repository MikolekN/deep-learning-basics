# DESCRIBE DATASET

import tensorflow as tf

from constants import DEBUG


def _describe_dataset(dataset_name, element_spec):
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


def describe_data(train_ds, val_ds):
    if DEBUG:
        _describe_dataset("Train dataset", train_ds.element_spec)
        _describe_dataset("Validation dataset", val_ds.element_spec)