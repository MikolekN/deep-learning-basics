import os

from class_to_number import class_names
import numpy as np

from keras.src.utils import load_img, img_to_array
from keras.src.applications.vgg16 import preprocess_input

from model import load_model_from_name

IMG_SIZE = (128, 128)
MODEL_NAME = "checkpoint_04.keras"
IMG_PATH = "./data/testing/kasia/C-12/2019_0721_171043_003 0564_0.jpg"

def extract_true_class(img_path):
    folder_name = os.path.basename(os.path.dirname(img_path))
    return folder_name

def load_and_preprocess_image(img_path, img_size):
    try:
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def predict_image(model, img_path, img_size):
    img_array = load_and_preprocess_image(img_path, img_size)
    if img_array is None:
        return None, None

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    return prediction, predicted_class


def single_inference(model_name, img_path, img_size):
    if not os.path.exists(img_path):
        print(f"Error: Image file '{img_path}' not found.")
        return

    model = load_model_from_name(model_name)

    prediction, predicted_class = predict_image(model, img_path, img_size)
    if prediction is None:
        return

    predicted_class_label = class_names[predicted_class]
    true_class = extract_true_class(img_path)

    print(f"Prediction Raw Output: {prediction}")
    print(f"Predicted Class Index: {predicted_class}")
    print(f"Predicted Class Label: {predicted_class_label}")
    print(f"Actual Class: {true_class}")

if __name__ == "__main__":
    single_inference(MODEL_NAME, IMG_PATH, IMG_SIZE)