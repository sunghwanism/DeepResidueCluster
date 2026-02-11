
import os
import sys
import logging
import argparse
import warnings
from pprint import pprint

# Ensure src is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import new modules
from src.utils.logger import get_logger, setup_logging
from src.data.datamodule import DeepResidueDataModule
from src.utils.functions import init_wandb, LoadConfig, set_seed

warnings.filterwarnings("ignore")
# Setup logging early
setup_logging(level=logging.INFO)
logger = get_logger("TrainScript")

def parse_args():
    parser = argparse.ArgumentParser(description='Run DGI model')
    parser.add_argument('--config_path', type=str, default='config/run.yaml', help='Path to the run configuration file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--wandb_key', type=str, default=None, help='Wandb API key')
    parser.add_argument('--entity_name', type=str, default='shmoon', help='Wandb entity name')
    parser.add_argument('--project_name', type=str, default='DeepResidueCluster', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default='DGI', help='Wandb run name')
    parser.add_argument('--wandb_run_id', type=str, default=None, help='Wandb run id')
    parser.add_argument('--load_pretrained', action='store_true', help='Load pretrained model')
    parser.add_argument('--nowandb', action='store_true', help='Do not use wandb')
    parser.add_argument('--use_aug', action='store_true', help='Use data augmentation')
    return parser.parse_args()

def main(args):
    # 1. Load Configuration
    config = LoadConfig(args)
    set_seed(config.SEED)
    
    # 2. Logger Setup
    logger.info("============================")
    logger.info("Initializing Training Pipeline")
    logger.info("============================")
    
    # 3. WandB Setup
    run_wandb = init_wandb(config)

    # 4. Data Module Initialization
    dm = DeepResidueDataModule(config)
    dm.setup()
    
    trainLoader = dm.train_dataloader()
    valLoader = dm.val_dataloader()
    testLoader = dm.test_dataloader()

    logger.info("##################")
    logger.info("Finish Loading DataLoader")
    logger.info(f"Train Batches: {len(trainLoader)}")
    logger.info("##################")
    
    # pprint(vars(config))

    if config.model == 'DGI':
        from src.models.DGI.execute import run_training
        run_training(config, trainLoader, valLoader, testLoader, run_wandb)
        
    elif config.model == 'pchk':
        from src.models.ProposedModel.execute import run_training
        run_training(config, trainLoader, valLoader, testLoader, run_wandb)

    else:
        logger.error(f"Unknown model: {config.model}")

    # Finish wandb run
    if run_wandb:
        run_wandb.finish()

if __name__ == '__main__':
    args = parse_args()
    main(args)
