import tensorflow as tf


def enable_gpu():
    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        print("No GPU detected. TensorFlow will run on CPU.")
        return

    print(f"GPUs detected: {gpus}")
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"Using GPU: {tf.config.get_visible_devices('GPU')}")
    except RuntimeError as e:
        print(f"Error while setting up GPU: {e}")
