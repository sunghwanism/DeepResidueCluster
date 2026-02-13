import os

import argparse

import pandas as pd
from CDSLoader import build_node_context_df



def get_parser():
    parser = argparse.ArgumentParser(description="Run CDS")
    parser.add_argument('--find_uniprot_file', required=True, type=str, help='py file with node_id list')
    parser.add_argument('--save_dir', required=True, type=str, help='directory to save cds context')

    return parser


def main(args):
    from args.find_uniprot_file import NODE_IDS
    
    result_df = build_node_context_df()
    result_df.to_csv(os.path.join(args.save_dir, 'cds_context.csv'), index=False)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)