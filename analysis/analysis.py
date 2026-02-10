import os
import sys
sys.path.append('../')

from typing import Union, List

import ast
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from bioservices import UniProt
from collections import Counter


def make_bin_cols(
    df: pd.DataFrame, 
    gen_col_name: str, 
    q_bin: Union[int, List[float]], 
    method: str = 'q'
) -> pd.DataFrame:
    """
    Bins mutability data and plots the distribution.
    method: 'bin' (equal width) or 'q' (quantile based)
    """
    if method == 'bin':
        df[gen_col_name] = pd.cut(df['mutability'], bins=q_bin, labels=False, include_lowest=True)
    elif method == 'q':
        df[gen_col_name] = pd.qcut(df['mutability'], q=q_bin, labels=False, duplicates='drop')
    
    # # Visualization
    # counts = df[gen_col_name].value_counts().sort_index()

    # plt.figure(figsize=(20, 3))
    # ax = sns.barplot(x=counts.index, y=counts.values, palette='viridis')

    # # Annotate with commas
    # for p in ax.patches:
    #     height = p.get_height()
    #     ax.annotate(f'{int(height):,}', 
    #                 (p.get_x() + p.get_width() / 2., height), 
    #                 ha='center', va='center', 
    #                 xytext=(0, 9), textcoords='offset points', fontsize=8)

    # plt.title('Mutability Bin Distribution', fontsize=15)
    # plt.xlabel('Mutability Bin', fontsize=12)
    # plt.ylabel('Count', fontsize=12)
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()
    
    return df


def get_gene_name(df, col_name='unique_uniprot'):
    uniprot_sets = df[col_name].dropna()
    unique_set = set()

    for uniprots in uniprot_sets:
        try:
            _list = ast.literal_eval(uniprots)
            unique_set.update(_list)
        except (ValueError, SyntaxError):
            continue

    if not unique_set:
        return {}

    u = UniProt()
    res = u.mapping(fr="UniProtKB_AC-ID", to="Gene_Name", query=list(unique_set))
    mapping_df = pd.DataFrame(res['results'])
    mapping_df = mapping_df.rename(columns={'from': 'uniprot_id', 'to': 'gene_name'})

    return mapping_df


def get_unique_uniprot(df, col_name='unique_uniprot'):
    uniprot_sets = df[col_name].dropna()
    unique_set = set()

    for uniprots in uniprot_sets:
        try:
            _list = ast.literal_eval(uniprots)
            unique_set.update(_list)
        except (ValueError, SyntaxError):
            continue

    if not unique_set:
        return {}

    return unique_set

def get_gene(row, gene_dict):
    try:
        uniprot_list = ast.literal_eval(row['unique_uniprot'])
        gene_list = [gene_dict.get(uniprot, uniprot) for uniprot in uniprot_list]
        return gene_list
    except (ValueError, SyntaxError, TypeError):
        return []


def get_counts(df, col_name):
    uniprot_sets = df[col_name].dropna()
    
    all_uniprots = []

    for uniprots in uniprot_sets:
        try:
            _list = ast.literal_eval(uniprots)
        except:
            _list = uniprots
        all_uniprots.extend(_list)
        
    uniprot_counts = dict(Counter(all_uniprots))

    return uniprot_counts