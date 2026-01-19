import gc
import wandb

import torch

def init_wandb(config, key=None):
    if not config.nowandb:
        if key is not None:
            wandb.login(key=key)
    
        if config.load_pretrained:
            wandb.init(project=config.project_name,
                        entity=config.entity_name,
                        id=config.wandb_run_id,
                        resume='must',
                        reinit=True,
                        name=config.wandb_run_name)

        else:
            wandb.init(project=config.project_name,
                        entity=config.entity_name,
                        name=config.wandb_run_name,
                        config=config)

        config.wandb_url = wandb.run.get_url()


def print_time(training_time):
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    formatted_time = f"{hours}:{minutes:02}:{seconds:02}"

    return formatted_time

def clean_the_memory():
    gc.collect()
    torch.cuda.empty_cache()