import os

import ast
import pickle
import argparse
import pandas as pd

from analysis import make_bin_cols
from enrich import analyze_all_clusters_parallel

import warnings
warnings.filterwarnings("ignore")

def main(config):
    
    all_node_df = pd.read_csv(config.node_path)
    all_node_df['ensmbl_id'] = all_node_df['ensmbl_id'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    all_node_df['is_mut'] = (all_node_df['total_mutations_count'] > 0).astype(int)
    all_node_df.reset_index(drop=True, inplace=True)
    all_node_df = all_node_df[~all_node_df['cds_contexts'].isna()].copy()
    
    all_node_df = make_bin_cols(all_node_df, 'bin_mutability', q_bin=config.bin, method='bin')
    clusterPATH = os.path.join(config.cluster_path, f'MCL_{config.params}.pkl')

    with open(clusterPATH, 'rb') as f:
        save_dict = pickle.load(f)
    
    print(save_dict['params'])
    clusters = save_dict['clusters']

    result_df = analyze_all_clusters_parallel(
            df=all_node_df,
            clusters_dict=clusters,
            min_mut=config.MIN_MUTATIONS,
            n_permutations=config.N_PERMS,
            n_jobs=config.N_CORES,
            stratify=config.stratify,
            bin_col='bin_mutability',
            PDBMatching=False
        )

    SAVEFILE = os.path.join(config.savepath, f'MCL_{config.params}_{config.stratify}_nP1m_bin{config.bin}.csv')
    result_df.to_csv(SAVEFILE, index=False)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster_path', type=str, help='Path to MCL clusters')
    parser.add_argument('--savepath', type=str, help='Path to save results')
    parser.add_argument('--node_path', type=str, help='Path to node mutation data')

    parser.add_argument('--params', type=str, help='ex. e5i3p08')
    parser.add_argument('--bin', type=int, help='Number of bins for mutability stratification')
    parser.add_argument('--stratify', type=str, help='Stratification method: mut, res, mut+res')

    parser.add_argument('--MIN_MUTATIONS', type=int, default=0, help='Minimum mutations in cluster to be tested')
    parser.add_argument('--N_PERMS', type=int, default=100_000, help='Number of permutations for enrichment test')
    parser.add_argument('--N_CORES', type=int, default=-1, help='Number of cores for parallel processing')

    config = parser.parse_args()
    main(config)