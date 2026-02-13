import os

import argparse

import pandas as pd
from CDSLoader import build_node_context_df

from saved_all_node_ids_for_cds import NODE_IDS as ALL_NODE_IDS
from saved_new_node_ids_for_cds import NODE_IDS as NEW_NODE_IDS

def get_parser():
    parser = argparse.ArgumentParser(description="Run CDS")
    parser.add_argument('--find_node_type', default='all', type=str, help='all or new')
    parser.add_argument('--save_dir', required=True, type=str, help='directory to save cds context')

    return parser


def main(args):    
    if not os.path.exists(args.save_dir):
        raise ValueError('save_dir does not exist')

    if args.find_node_type == 'all':
        NODE_IDS = ALL_NODE_IDS
        print(f"Find all node cds context: {len(NODE_IDS)}")
    elif args.find_node_type == 'new':
        NODE_IDS = NEW_NODE_IDS
        print(f"Find new node cds context: {len(NODE_IDS)}")
    else:
        raise ValueError('find_node_type must be all or new')
    
    result_df = build_node_context_df(NODE_IDS)
    result_df.to_csv(os.path.join(args.save_dir, f'{args.find_node_type}_nodes_cds_context.csv'), index=False)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)