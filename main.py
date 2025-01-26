from constants import INPUT_SHAPE, NUM_CLASSES, CONFIG
from gpu_setup import enable_gpu
from load_dataset import load_training_dataset
from model import create_model, train_model, save_model
from wandb_context import wandb_session


enable_gpu()

train_ds, val_ds = load_training_dataset(supplemented=True)

with wandb_session(CONFIG) as wandb_config:
    model = create_model(INPUT_SHAPE, NUM_CLASSES)
    history = train_model(model, train_ds, val_ds)
    save_model(model)
