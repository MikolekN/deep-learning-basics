import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from class_to_number import class_names
from load_dataset import load_dataset
from model import create_model, train_model, save_model, predict_image

print("TensorFlow version:", tf.__version__)

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPUs available: {physical_devices}")
else:
    print("No GPU detected. TensorFlow will run on CPU.")

if physical_devices:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')  # Use the first GPU
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_ds, val_ds = load_dataset()
model = create_model()
history = train_model(model, train_ds, val_ds)
save_model(model)
prediction = predict_image(model, "data/testing/our/A-3/A-3.jpg")
print(f"Prediction: {prediction}")
predicted_class = np.argmax(prediction, axis=1)
print(f"Predicted class: {predicted_class[0]}")
predicted_class_label = class_names[predicted_class[0]]
print(f"Predicted class label: {predicted_class_label}")

# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.title('Model Accuracy')
# plt.show()
#
# # Plot loss
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('Model Loss')
# plt.show()
