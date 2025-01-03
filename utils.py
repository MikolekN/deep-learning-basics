from datetime import datetime


def create_run_name(config) -> str:
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{current_datetime}?opt={config['optimizer']}-lr={config['learning_rate']:.3f}-bs={config['batch_size']}-ep={config['epochs']}"

    return name


def create_checkpoint_name() -> str:
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{current_datetime}"

    return name


def create_model_name() -> str:
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{current_datetime}"

    return name
