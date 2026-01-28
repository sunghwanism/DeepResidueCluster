import os
import time
from pprint import pprint
import pickle
import argparse as args

import pandas as pd
import numpy as np
import networkx as nx

from utils import time_calc
from Calculator import calcuator
from centralityConfig import centrality_measures

USECOLS = ['unique_patients_count', 'total_mutations_count', 'unique_mutation_types_count']

def get_parser():
    parser = args.ArgumentParser(description="Calculate graph centrality measures.")
    
    # Basic Setting
    parser.add_argument('--node_path', type=str, required=True, help='Path to node attribute data (CSV file)')
    parser.add_argument('--graph_path', type=str, required=True, help='Path to graph data (pickle file)')
    parser.add_argument('--savepath', type=str, required=True, help='Path to save results')
    parser.add_argument('--logpath', type=str, help='Path to save logs')
    
    # Calculation
    parser.add_argument('--measure', type=str, nargs='+', required=True, help='Measure(s) to calculate. Use "all" for all available measures.')
    parser.add_argument('--param_path', type=str, help='Path to calculation parameters (centralityConfig.py)')
    
    return parser

def set_device(measures):
    """Decides whether to use CPU or GPU based on available measures and hardware."""
    use_gpu = any(m in CU_NX_AVAILABLE for m in measures)
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                print("CUDA is available. Attempting to use GPU.")
                os.environ['NX_CUGRAPH_AUTOCONFIGURE'] = 'True'
                return 'cuda'
        except ImportError:
            print("PyTorch not found. Cannot use GPU.")
    
    print("Using CPU for calculations.")
    return 'cpu'

def main(config):
    start = time.time()
    print("=" * 50)
    print("Starting centrality calculation process...")
    pprint(config.__dict__)
    print("=" * 50)
    
    # The user's environment has install errors with GPU libs, so we force CPU.
    # device = set_device(config.measure)
    device = 'cpu'
    
    print(f"Using device: {device}")
    print('[NetworkX Configuration]')
    pprint(nx.config)
    print("=" * 50)

    # 1. Load data
    print("Loading graph and attributes...")
    node_att_df = pd.read_csv(config.node_path)
    with open(config.graph_path, 'rb') as f:
        G = pickle.load(f)
    
    print("[Graph Info]")
    print(G)
    print("=" * 50)
    
    # 2. Calculate
    print(f"Calculating measures: {config.measure}")
    Complete = calcuator(G, config, device, centrality_measures)
    
    end = time.time()
    hours, minutes, seconds = time_calc(start, end)

    if Complete == "Success":
        print(f"All processes completed successfully.")
    else:
        print("An error occurred during calculation.")

    print(f"Total processing time: {hours}:{minutes:02d}:{seconds:02d}")
    print("=" * 50)

if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_args()
    
    if not os.path.exists(config.savepath):
        os.makedirs(config.savepath)
        
    main(config)