import torch
import pandas as pd
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


def nx_to_pyg_data(G, node_features_df=None, label_col=None, feature_cols=None):
    """
    Convert a NetworkX graph to a PyTorch Geometric Data object.
    
    Args:
        G (nx.Graph): NetworkX graph
        node_features_df (pd.DataFrame, optional): DataFrame with node features.
            Index should be node IDs matching the graph.
        label_col (str, optional): Column name in node_features_df to use as labels.
        feature_cols (list, optional): List of column names to use as features.
            If None, uses all columns except label_col.
    
    Returns:
        Data: PyTorch Geometric Data object
    """
    # Convert graph structure to PyG format
    data = from_networkx(G)
    
    # Get node list in order
    nodes = list(G.nodes())
    num_nodes = len(nodes)
    
    # Create node mapping (node_id -> index)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    # Handle node features
    if node_features_df is not None:
        # Ensure DataFrame index matches nodes
        if not all(node in node_features_df.index for node in nodes):
            raise ValueError("Not all graph nodes are present in node_features_df index")
        
        # Reorder DataFrame to match node order
        df_ordered = node_features_df.loc[nodes]
        
        # Extract labels if specified
        if label_col is not None:
            if label_col not in df_ordered.columns:
                raise ValueError(f"Label column '{label_col}' not found in DataFrame")
            labels = df_ordered[label_col].values
            data.y = torch.tensor(labels, dtype=torch.long)
        
        # Extract features
        if feature_cols is None:
            # Use all columns except label_col
            feature_cols = [col for col in df_ordered.columns if col != label_col]
        
        if len(feature_cols) > 0:
            features = df_ordered[feature_cols].values
            data.x = torch.tensor(features, dtype=torch.float)
        else:
            # Use identity features if no feature columns
            data.x = torch.eye(num_nodes, dtype=torch.float)
    else:
        # No DataFrame provided, use identity features
        data.x = torch.eye(num_nodes, dtype=torch.float)
    
    return data


def create_dgi_training_data(data):
    """
    Prepare data for DGI training by creating structures for positive and negative samples.
    
    Args:
        data (Data): PyTorch Geometric Data object
    
    Returns:
        dict: Dictionary containing:
            - 'x': Node features
            - 'edge_index': Edge indices
            - 'num_nodes': Number of nodes
    """
    return {
        'x': data.x,
        'edge_index': data.edge_index,
        'num_nodes': data.x.size(0),
        'y': data.y if hasattr(data, 'y') else None
    }


def shuffle_node_features(x):
    """
    Shuffle node features for negative sampling in DGI.
    
    Args:
        x (torch.Tensor): Node features (N, F)
    
    Returns:
        torch.Tensor: Shuffled node features
    """
    idx = torch.randperm(x.size(0))
    return x[idx]