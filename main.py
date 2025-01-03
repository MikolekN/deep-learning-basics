import numpy as np
import tensorflow as tf

from class_to_number import class_names
from constants import INPUT_SHAPE, NUM_CLASSES
from load_dataset import load_dataset
from model import create_model, train_model, save_model, predict_image, _initialize, _uninitialize

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPUs available: {physical_devices}")
else:
    print("No GPU detected. TensorFlow will run on CPU.")

if physical_devices:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_ds, val_ds = load_dataset()

wandb_config = _initialize()
model = create_model(wandb_config, INPUT_SHAPE, NUM_CLASSES)
history = train_model(model, train_ds, val_ds)
save_model(model)
_uninitialize()

prediction = predict_image(model, "data/testing/our/A-3/A-3.jpg")
print(f"Prediction: {prediction}")
predicted_class = np.argmax(prediction, axis=1)
print(f"Predicted class: {predicted_class[0]}")
predicted_class_label = class_names[predicted_class[0]]
print(f"Predicted class label: {predicted_class_label}")

# plot_metric(history=history, metric_name='accuracy')
# plot_metric(history=history, metric_name='loss')