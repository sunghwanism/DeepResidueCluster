
import os
import sys
import pickle

import argparse
import warnings

# Ensure src is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.data.datamodule import DeepResidueDataModule
from src.utils.functions import LoadConfig, set_seed

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare and save Train/Val/Test datasets')
    parser.add_argument('--config_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), '../config/run.yaml'),
                        help='Path to the run configuration file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--use_aug', action='store_true', help='Use data augmentation')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save processed datasets (default: config.SAVEPATH/data)')

    # Unused but kept for LoadConfig compatibility
    parser.add_argument('--wandb_key', type=str, default=None)
    parser.add_argument('--entity_name', type=str, default='shmoon')
    parser.add_argument('--project_name', type=str, default='DeepResidueCluster')
    parser.add_argument('--wandb_run_name', type=str, default='DGI')
    parser.add_argument('--wandb_run_id', type=str, default=None)
    parser.add_argument('--load_pretrained', action='store_true')
    parser.add_argument('--nowandb', action='store_true')
    return parser.parse_args()


def save_dataset(data_list, filepath):
    """Save a list of PyG Data objects to a pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data_list, f)
    print(f"Saved {len(data_list)} graphs -> {filepath}")


def main(args):
    # 1. Load Configuration & Set Seed
    config = LoadConfig(args)
    set_seed(config.SEED)

    # 2. Determine output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(getattr(config, 'SAVEPATH', '.'), 'data')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("Starting Dataset Preparation")
    print(f"Output directory: {output_dir}")
    print(f"use_aug: {args.use_aug}")
    print("=" * 50)

    # 3. Build base datasets (without augmentation)
    config.use_aug = False
    dm = DeepResidueDataModule(config)
    dm.setup()

    save_dataset(dm.train_data, os.path.join(output_dir, 'train.pkl'))
    save_dataset(dm.val_data,   os.path.join(output_dir, 'val.pkl'))
    save_dataset(dm.test_data,  os.path.join(output_dir, 'test.pkl'))

    print(f"Base datasets saved — Train: {len(dm.train_data)}, "
          f"Val: {len(dm.val_data)}, Test: {len(dm.test_data)}")

    # 4. Build augmented train dataset (if requested)
    if args.use_aug:
        print("Building augmented training dataset...")
        config.use_aug = True
        dm_aug = DeepResidueDataModule(config)
        dm_aug.setup()

        save_dataset(dm_aug.train_data, os.path.join(output_dir, 'train_aug.pkl'))
        print(f"Augmented train dataset saved — "
              f"Train(aug): {len(dm_aug.train_data)} "
              f"(base: {len(dm.train_data)})")

    print("Dataset preparation complete.")


if __name__ == '__main__':
    args = parse_args()
    main(args)


#########################
# Debugging run script
#########################

# python scripts/runData.py \
#     --config_path config/run.yaml \
#     --project_name DeepResidueCluster \
#     --batch_size 256 \
#     --num_workers 4 \
#     --nowandb \
#     --output_dir . \
#     --use_aug