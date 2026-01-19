import os

import torch
import pandas as pd
import networkx as nx
import numpy as np

import pickle
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
    
    # 1. Convert graph structure and extract internal graph features
    data = from_networkx(G, group_node_attrs=graph_features)

    # 2. Handle Edge Weights
    if not use_edge_weight and hasattr(data, 'edge_weight'):
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

def loadGraph(path):
    with open(path, 'rb') as f:
        G = pickle.load(f)
    return G

def filtered_only_attributes(G, targets):
    resultG = G.copy()
    if isinstance(targets, str):
        targets = [targets]

    for n in resultG.nodes:
        for attr in list(resultG.nodes[n].keys()):
            if attr not in targets:
                resultG.nodes[n].pop(attr)
            
    return resultG

def merge_graph_attributes(rawG, config):
    """
    Merges node attributes from external pickle files into the main graph (rawG).
    Includes validation to ensure node consistency and attribute existence.
    """
    PATH = config.GraphAtt_PATH
    files = os.listdir(PATH)
    use_features = config.graph_features
    filtered_features = []

    # Get the set of nodes from the original graph for consistency checks
    raw_nodes = set(rawG.nodes())
    num_raw_nodes = rawG.number_of_nodes()

    for pkl in files:
        # 1. Filter files by naming convention
        if pkl.endswith('.pkl') and pkl.startswith('graph_with_'):
            # Extract internal attribute name from filename
            att_name = pkl.removeprefix("graph_with_").removesuffix(".pkl")
            
            # Normalize specific feature names (e.g., shortest_path_length)
            if 'shortest_path_length' in att_name:
                display_name = 'shortest_path_length'
            else:
                display_name = att_name

            # 2. Check if this feature is requested in config
            if display_name in use_features:
                try:
                    # Load the attribute graph (contains node features)
                    attG = loadGraph(os.path.join(PATH, pkl))
                    
                except Exception as e:
                    print(f"Error loading {pkl}: {e}")
                    continue
                
                # --- Validation Step 1: Structural Consistency ---
                # Check if node counts and node IDs match perfectly
                att_nodes = set(attG.nodes())
                if num_raw_nodes == attG.number_of_nodes() and att_nodes == raw_nodes:
                    
                    # --- Validation Step 2: Attribute Existence ---
                    # Sample the first node to verify the attribute key exists in attG
                    sample_node = next(iter(attG.nodes()))
                    actual_keys = attG.nodes[sample_node].keys()
                    
                    if not actual_keys:
                        print(f"Warning: No attributes found in nodes of {pkl}")
                        continue

                    # Merge attributes from attG into rawG
                    for node, attrs in attG.nodes(data=True):
                        # Update rawG node dictionary with new attributes
                        rawG.nodes[node].update(attrs)
                    
                    filtered_features.append(display_name)
                    print(f"Successfully merged feature: [{display_name}] from {pkl}")
                else:
                    print(f"Validation Failed for {pkl}: Node mismatch (Count or IDs)")

    # Ensure uniqueness in the feature list
    filtered_features = list(set(filtered_features))
    finalG = filtered_only_attributes(rawG, targets=filtered_features)
    
    # --- Final Validation Step 3: Global Feature Completeness ---
    final_verified_features = []
    for feat in filtered_features:
        missing_nodes = [n for n, d in finalG.nodes(data=True) if feat not in d]
        if not missing_nodes:
            final_verified_features.append(feat)
        else:
            print(f"CRITICAL: Feature '{feat}' is missing in {len(missing_nodes)} nodes! Dropping from list.")

    print("Final Filtered Features for PyG conversion: ", final_verified_features)
    print('============================ '*2)
    
    # Update config or return the list alongside the graph if necessary
    return finalG