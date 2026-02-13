import os

import networkx as nx
import pandas as pd

def filter_only_nucleosome_related_connection(edge_df, excl_pdb_df, incl_uniprot_df):
    past_len = len(edge_df)

    excl_pdb_list = excl_pdb_df['0'].str.lower().unique().tolist()
    edge_df = edge_df[~edge_df['pdb_code'].isin(excl_pdb_list)].reset_index(drop=True)
    
    print(f"Remove excl pdb Connection: [{past_len} -> {len(edge_df)}] -> Removed {past_len - len(edge_df)} Connection")

    past_len = len(edge_df)
    incl_list = incl_uniprot_df['uniprot'].str.lower().unique().tolist()

    mask = (edge_df['remove_homo_uniprot1'].isin(incl_list)) & \
           (edge_df['remove_homo_uniprot2'].isin(incl_list))
    
    edge_df = edge_df[mask].reset_index(drop=True)

    print(f"Filter only nucleosome related Connection: [{past_len} -> {len(edge_df)}] -> Removed {past_len - len(edge_df)} Connection")
    
    return edge_df

def remove_ubq_related_connection(edge_df, remove_ubq_list):
    past_len = len(edge_df)
    edge_df = edge_df[~edge_df['uniprot1'].isin(remove_ubq_list) & ~edge_df['uniprot2'].isin(remove_ubq_list)]

    print(f"Remove ubq related Connection: [{past_len} -> {len(edge_df)}] -> Removed {past_len - len(edge_df)} Connection")
    
    return edge_df


def generate_nodeid_and_only_uniprot(edge_df):
    edge_df['nodeid_1'] = (edge_df['uniprot1'].str.replace('_', '-', regex=False).astype(str) + "_" +
    edge_df['pdb_auth_resi1'].astype(str) + "_" +
    edge_df['res3n1'].astype(str)).str.lower()

    edge_df['nodeid_2'] = (edge_df['uniprot2'].str.replace('_', '-', regex=False).astype(str) + "_" +
    edge_df['pdb_auth_resi2'].astype(str) + "_" +
    edge_df['res3n2'].astype(str)).str.lower()

    edge_df['remove_homo_uniprot1'] = edge_df['uniprot1'].str.split("_").str[0]
    edge_df['remove_homo_uniprot2'] = edge_df['uniprot2'].str.split("_").str[0]

    return edge_df


def filter_only_have_cds_context(edge_df, bmr_df):

    past_len = len(edge_df)
    incl_list = bmr_df['node_id'].tolist()

    mask = (edge_df['nodeid_1'].isin(incl_list)) & \
           (edge_df['nodeid_2'].isin(incl_list))
    
    edge_df = edge_df[mask].reset_index(drop=True)

    print(f"Filter only have cds context Connection: [{past_len} -> {len(edge_df)}] -> Removed {past_len - len(edge_df)} Connection")
    
    return edge_df


def get_unique_pid_from_edge_df(edge_df):

    uni1_pid = edge_df['nodeid_1'].unique().tolist()
    uni2_pid = edge_df['nodeid_2'].unique().tolist()

    return list(set(uni1_pid + uni2_pid))

def remove_negative_and_zero_position_node(edge_df):
    past_len = len(edge_df)
    edge_df = edge_df[(edge_df['pdb_auth_resi1'].astype(int) > 0) & (edge_df['pdb_auth_resi2'].astype(int) > 0)].reset_index(drop=True)
    print(f"Remove negative and zero position node Connection: [{past_len} -> {len(edge_df)}] -> Removed {past_len - len(edge_df)} Connection")
    return edge_df