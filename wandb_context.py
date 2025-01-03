from contextlib import contextmanager
import wandb
from utils import create_run_name


@contextmanager
def wandb_session(config):
    hyperparams = dict(config)
    try:
        wandb.init(project="polish-road-signs-classification",
                   name=create_run_name(config=hyperparams),
                   config=hyperparams)
        yield wandb.config
    finally:
        wandb.finish(exit_code=0)
