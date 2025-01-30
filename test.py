from constants import DEBUG
from load_dataset import load_testing_dataset
from model import load_model_from_name
from report import generate_classification_report, calculate_class_accuracy


def report(model, test_ds):
    generate_classification_report(model, test_ds)
    calculate_class_accuracy(model, test_ds)

if __name__ == "__main__":
    BEST_MODEL_NAME = 'checkpoint_04.keras'
    model = load_model_from_name(BEST_MODEL_NAME)
    test_ds = load_testing_dataset()

    test_metrics = model.evaluate(test_ds, verbose=DEBUG)

    metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
    for metric_name, value in zip(metric_names, test_metrics):
        print(f"{metric_name}: {value:.4f}")

    report(model, test_ds)