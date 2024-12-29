import os


# --- DEBUGGING --- #
DEBUG = False

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


# --- IMAGE PARAMETERS --- #

# Resize training images to a uniform size
IMG_SIZE = (128, 128)

# Batch of images to be processes at the same time
IMG_BATCH_SIZE = 32

