from sklearn.metrics import classification_report
import numpy as np

from class_to_number import class_names


def generate_classification_report(model, test_ds):
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        predictions = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(predictions.argmax(axis=1))

    labels = list(range(len(class_names)))

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        labels=labels,
        zero_division=0
    )
    print(report)


def calculate_class_accuracy(model, test_ds):
    correct = np.zeros(len(class_names))
    total = np.zeros(len(class_names))

    for images, labels in test_ds:
        predictions = model.predict(images).argmax(axis=1)
        for label, prediction in zip(labels.numpy(), predictions):
            total[label] += 1
            if label == prediction:
                correct[label] += 1

    for i, class_name in enumerate(class_names):
        accuracy = correct[i] / total[i] if total[i] > 0 else 0
        print(f"Class: {class_name}, Accuracy: {accuracy:.2f}")