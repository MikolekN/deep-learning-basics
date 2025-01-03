import numpy as np

from class_to_number import class_names
from constants import INPUT_SHAPE, NUM_CLASSES, CONFIG
from gpu_setup import enable_gpu
from load_dataset import load_dataset
from model import create_model, train_model, save_model, predict_image
from wandb_context import wandb_session


enable_gpu()

train_ds, val_ds = load_dataset()

with wandb_session(CONFIG) as wandb_config:
    model = create_model(INPUT_SHAPE, NUM_CLASSES)
    history = train_model(model, train_ds, val_ds)
    save_model(model)

prediction = predict_image(model, "data/testing/our/A-3/A-3.jpg")
print(f"Prediction: {prediction}")
predicted_class = np.argmax(prediction, axis=1)
print(f"Predicted class: {predicted_class[0]}")
predicted_class_label = class_names[predicted_class[0]]
print(f"Predicted class label: {predicted_class_label}")
