import numpy as np
from class_to_number import class_names
from constants import INPUT_SHAPE, NUM_CLASSES, CONFIG, SWEEP_CONFIG, SWEEP_RUN_COUNT
from gpu_setup import enable_gpu
from load_dataset import load_training_dataset
from model import create_model, train_model, save_model
from wandb_context import wandb_session
import wandb


SWEEPS = True

enable_gpu()

train_ds, val_ds = load_training_dataset(supplemented=True)

def run():
    with wandb_session(CONFIG) as wandb_config:
        model = create_model(INPUT_SHAPE, NUM_CLASSES)
        history = train_model(model, train_ds, val_ds)
        save_model(model)
        return model


if SWEEPS:
    sweep_id = wandb.sweep(sweep=SWEEP_CONFIG, project='polish-road-signs-classification')
    wandb.agent(sweep_id, function=run, count=SWEEP_RUN_COUNT, project='polish-road-signs-classification')
else:
    model = run()

# Get best run parameters
# best_run = sweep.best_run()
# best_parameters = best_run.config
