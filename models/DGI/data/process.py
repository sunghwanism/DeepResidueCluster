import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import torch
import torch.nn as nn


def load_graph_from_nx(G, features_df=None, label_col=None, feature_cols=None):
    """
    Load graph from NetworkX and optionally merge with DataFrame features.
    
    Args:
        G (nx.Graph): NetworkX graph
        features_df (pd.DataFrame, optional): DataFrame with node features
        label_col (str, optional): Column name for labels
        feature_cols (list, optional): Column names for features
    
    Returns:
        Data: PyG Data object
    """
    from .graph_utils import nx_to_pyg_data
    
    return nx_to_pyg_data(G, features_df, label_col, feature_cols)
