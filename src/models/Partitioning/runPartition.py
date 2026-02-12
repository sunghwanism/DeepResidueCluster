import os
import argparse

import pickle
import pandas as pd

from partition import remove_all_repulsive_edge, change_cluster_from_threshold


def getParser():
    args = argparse.ArgumentParser(description='Partitioning Parser')
    args.add_argument('--graphpath', type=str, required=True) # Need to check edge weight Todo

    args.add_argument('--threshold', type=int, required=True)
    args.add_argument('--all_repulsive', type=bool, default=False)

    args.add_argument('--save_prefix', type=str, required=True)
    args.add_argument('--savepath', type=str, default='result')

    return args

def main(config):
    wG = load_grpah(config.graphpath)

    if wG.edges.data('weight') is None:
        raise ValueError("Graph must have edge weights")

    if config.all_repulsive:
        remove_all_repulsive = remove_all_repulsive_edge(wG)
        all_rep_clusters = change_cluster_from_threshold(remove_all_repulsive)
        save_file_path = os.path.join(config.savepath, f'{config.save_prefix}_partition_all_repulsive.pkl')
        print(f"Saving all repulsive clusters to {save_file_path}")
        with open(save_file_path, 'wb') as f:
            pickle.dump(all_rep_clusters, f)

    # return connected components List
    result_cc_list = remove_repulsive_with_threshold(wG, config.threshold)

    # return cluster dict {cluster_id: [node1, node2, ...]}
    result_clusers = change_cluster_from_threshold(result_cc_list)
    save_file_path = os.path.join(config.savepath, f'{config.save_prefix}_partition_th{config.threshold}.pkl')
    print(f"Saving clusters to {save_file_path}")
    with open(save_file_path, 'wb') as f:
        pickle.dump(result_clusers, f)
    

if __name__ == '__main__':

    args = getParser().parse_args()
    main(args)
