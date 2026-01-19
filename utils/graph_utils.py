import torch
import pandas as pd
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

def nx_to_pyg_data(G, node_features_df=None, label_col=None, 
                   graph_features=None, 
                   table_features=None,
                   add_constant_feature=True, 
                   use_edge_weight=False):
    """
    Converts a NetworkX graph to a PyTorch Geometric Data object, 
    merging features from both the graph attributes and an external DataFrame.

    Args:
        G (nx.Graph): The input NetworkX graph.
        node_features_df (pd.DataFrame, optional): DataFrame containing external node features.
            The index must contain the node IDs from G.
        label_col (str, optional): The column name in node_features_df to use as labels (y).
        graph_features (list, optional): List of node attribute keys in G to use as features.
            (e.g., ['degree', 'closeness'])
        table_features (list, optional): List of column names in node_features_df to use as features.
            (e.g., ['shortpath'])
        use_edge_weight (bool, optional): If False, removes edge weights even if they exist in G.

    Returns:
        Data: A PyTorch Geometric Data object with merged features (x) and labels (y).
    """
    print("Start Convert NetworkX to PyG Data")
    print("============================ "*2)
    
    # 1. Convert graph structure and extract internal graph features
    data = from_networkx(G, group_node_attrs=graph_features)

    # 2. Handle Edge Weights
    if not use_edge_weight and hasattr(data, 'edge_weight'):
        print("Not Use Edge Weight")
        del data.edge_weight
    
    # 3. Align Node Order (Crucial for mapping DataFrame to Graph)
    nodes = list(G.nodes())
    num_nodes = len(nodes)
    
    # 4. Process DataFrame Features (and merge with Graph Features)
    if node_features_df is not None:
        # Verify that all graph nodes exist in the DataFrame
        if not all(node in node_features_df.index for node in nodes):
            raise ValueError("Error: Not all nodes in the graph are present in the node_features_df index.")
        
        # Reorder DataFrame to match the order of nodes in the graph
        df_ordered = node_features_df.loc[nodes]
        
        # (A) Extract Labels (y)
        if label_col is not None:
            if label_col not in df_ordered.columns:
                raise ValueError(f"Error: Label column '{label_col}' not found in DataFrame.")
            
            # Convert to tensor (Assuming Long type for classification tasks)
            labels = df_ordered[label_col].values
            data.y = torch.tensor(labels, dtype=torch.long) # for classification tasks
        
        # (B) Extract Tabular Features
        # If table_features is not provided, use all columns except the label column
        features_to_use = table_features
        if features_to_use is None:
            features_to_use = [col for col in df_ordered.columns if col != label_col]
        
        if len(features_to_use) > 0:
            table_x = torch.tensor(df_ordered[features_to_use].values, dtype=torch.float)
            
            # [Core Logic] Merge Graph Features and Table Features
            # If data.x already exists (from graph_features), concatenate them along dim 1.
            if hasattr(data, 'x') and data.x is not None:
                data.x = torch.cat([data.x, table_x], dim=1)
            else:
                # If only table features exist, assign them directly
                data.x = table_x
    
    if not hasattr(data, 'x') or data.x is None:
        raise ValueError("Error: No features provided. Please provide either graph features or table features.")

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