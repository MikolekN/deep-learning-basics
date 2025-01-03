import matplotlib.pyplot as plt

def plot_metric(history, metric_name):
    plt.plot(history.history[metric_name], label='Train Loss')
    plt.plot(history.history['val_' + metric_name], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.title('Model ' + metric_name)
    plt.show()
