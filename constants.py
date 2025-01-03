import os


# --- DEBUGGING --- #
from collections import namedtuple

from class_to_number import class_names

DEBUG = True


# --- DATA DIRECTORIES --- #

# Main data directory
_DATA_DIR = "data"
_TRAINING_DATA_DIR = "training"
_TESTING_DATA_DIR = "testing"

# Main training data directory (training data from 'polish traffic signs dataset' by Kasia12345)
TRAINING_DATA_DIR = os.path.join(_DATA_DIR, _TRAINING_DATA_DIR, "kasia")

# Supplement training data directory (training data from 'Polish Traffic Signs Dataset' by ChrisKJM)
TRAINING_DATA_SUPPLEMENT_DIR = os.path.join(_DATA_DIR, _TRAINING_DATA_DIR, "chris")

# Main testing data directory (testing data from 'polish traffic signs dataset' by Kasia12345)
TEST_DATA_DIR = os.path.join(_DATA_DIR, _TESTING_DATA_DIR, "kasia")

# Secondary testing data directory (testing data gathered on our own)
TRAINING_DATA_SECONDARY_DIR = os.path.join(_DATA_DIR, _TESTING_DATA_DIR, "our")

# Main model directory
_MODEL_DIR = "models"

# Main saved model directory
SAVED_MODEL_DIR = os.path.join(_MODEL_DIR)

# Directory of model ids
SAVED_MODEL_IDS = "model_id.txt"


# --- TRAINING PARAMETERS --- #

EPOCHS = 5

BATCH_SIZE = 64

CONFIG = {
    'filters': 60,                      # Number of filters for the first layer
    'kernel_1': (5, 5),                 # Kernel size for the first set of Conv2D layers
    'kernel_2': (3, 3),                 # Kernel size for the second set of Conv2D layers
    'padding': 'valid',                 # Padding type
    'pooling': (2, 2),                  # Pooling size for MaxPooling2D
    'learning_rate': 0.001,             # Learning rate for the Adam optimizer
    'wd': 0.0,                          # Weight decay (optional, used for regularization)
    'learning_rate_schedule': 'RLR',    # Learning rate schedule: ReduceLROnPlateau (RLR), cyclic, step decay
    'optimizer': 'adam',                # Optimizer type (e.g., Adam, RMSProp)
    'dense_units': 500,                 # Number of units in the dense layer
    'activation': 'relu',               # Activation function (e.g., relu, elu, LeakyReLU)
    'dropout': 0.5,                     # Dropout rate for intermediate layers, can be different value than dropout_f, place for experiments
    'dropout_f': 0.5,                   # Dropout rate for fully connected layers
    'batch_size': BATCH_SIZE,           # Batch size for training
    'epochs': EPOCHS                    # Number of epochs for training
}

# --- IMAGE PARAMETERS --- #

# Resize training images to a uniform size
IMG_SIZE = (128, 128)

# Batch of images to be processes at the same time
IMG_BATCH_SIZE = 32

ImageShape = namedtuple("ImageShape", ["height", "width", "channels"])

INPUT_SHAPE = ImageShape(128, 128, 3)

NUM_CLASSES = len(class_names)

# TODO
IMG_CHANNEL_NUMBER = 3
