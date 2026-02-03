
import os
import sys

import yaml
import argparse
from pprint import pprint


from data.process import LoadDataset, getDataLoader
from utils.functions import init_wandb, LoadConfig, set_seed

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Run DGI model')
    parser.add_argument('--config_path', type=str, default=os.path.join(os.path.dirname(__file__), '../config/run.yaml'), help='Path to the run configuration file')
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
    config = LoadConfig(args)
    set_seed(config.SEED)
    run_wandb = init_wandb(config)

    print('============================'*2)
    # Load Data
    train, test, val = LoadDataset(config, only_test=False, clear_att_in_orginG=True)
    trainLoader = getDataLoader(train, config)
    valLoader = getDataLoader(val, config, test=True) # test=True makes shuffle=False
    testLoader = getDataLoader(test, config, test=True) # test=True makes shuffle=False

    print("##################"*3)
    print("Finish Loading DataLoader")
    print("Train", len(train), "Val", len(val), "Test", len(test))
    print("##################"*3)
    pprint(config)
    print("##################"*3)

    if config.model == 'FFN': # Based on pretrained model
        from model.Classifier.FFN import FFNClassifier, TrainFFN
        from models.Classifier.execute import run_downstream
        clf_model = FFNClassifier(in_ft=config.emb_dim,
                                  out_ft_list=config.ds_out_ft_list,
                                  activation=config.ds_activation, 
                                  drop_prob=config.ds_drop_prob, n_cls=config.n_cls)


        run_downstream(config, clf_model, trainLoader, valLoader, testLoader)

        
        
    elif config.model == 'GAT':
        pass

    elif config.model == 'node2vec':
        pass

    else:
        raise ValueError(f"Unknown model: {config.model} || Select from ['DGI','FFN','GAT','node2vec']")


    

    # Finish wandb run
    if run_wandb:
        run_wandb.finish()

if __name__ == '__main__':
    args = parse_args()
    main(args)
