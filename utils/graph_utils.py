import os

import torch
import pandas as pd
import networkx as nx
import numpy as np

import pickle
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from utils.table_utils import EncodeFeatures

def nx_to_pyg_data(G, node_features_df=None, label_col=None, 
                   graph_features=None, 
                   table_features=None,
                   add_constant_feature=True, 
                   use_edge_weight=False,
                   config=None,
                   verbose=False):
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
    
    feature_names = []
    if graph_features:
        feature_names.extend([f"[Graph] {f}" for f in graph_features])

    # 2. Handle Edge Weights
    if not use_edge_weight and hasattr(data, 'edge_weight'):
        del data.edge_weight
    
    # 3. Align Node Order
    nodes = list(G.nodes())
    
    # 4. Process DataFrame Features
    if node_features_df is not None:
        if not all(node in node_features_df.index for node in nodes):
            raise ValueError("Error: Not all nodes in the graph are present in the node_features_df index.")
        
        df_ordered = node_features_df.loc[nodes].copy()
        
        if label_col is not None:
            labels = df_ordered[label_col].values
            data.y = torch.tensor(labels, dtype=torch.long)
        
        features_to_use = table_features.copy()
        if features_to_use is None:
            features_to_use = [col for col in df_ordered.columns if col != label_col]
        
        if len(features_to_use) > 0:
            result_df, category_feat, numerical_feat = EncodeFeatures(df_ordered, features_to_use, verbose)
            split_features = [col for col in features_to_use if col not in category_feat]
            split_features.extend(numerical_feat)
            if 'ptms_mapped' in features_to_use:
                split_features.remove('ptms_mapped')
            
            # Record numerical table features
            feature_names.extend([f"[Table] {f}" for f in split_features])
            table_x = torch.tensor(result_df[split_features].values, dtype=torch.float)

            if hasattr(data, 'x') and data.x is not None:
                data.x = torch.cat([data.x, table_x], dim=1)
            else:
                data.x = table_x

            if len(category_feat) > 0:
                cat_tensor = torch.tensor(result_df[category_feat].values, dtype=torch.long)
                if hasattr(data, 'x_cat') and data.x_cat is not None:
                    data.x_cat = torch.cat([data.x_cat, cat_tensor], dim=1)
                else:
                    data.x_cat = cat_tensor

    if add_constant_feature:
        num_nodes = data.x.size(0)
        constant_x = torch.ones((num_nodes, 1), dtype=torch.float)
        data.x = torch.cat([data.x, constant_x], dim=1)
        feature_names.append("[Extra] Constant Feature (1.0)")

    # [Feature Logging] Show feature to index mapping ONCE
    if verbose:
        print("\n" + "="*60)
        print(f"{'INDEX':<10} | {'SOURCE':<10} | {'FEATURE NAME'}")
        print("-"*60)
        for i, full_name in enumerate(feature_names):
            source, name = full_name.split('] ', 1)
            source = source.replace('[', '')
            print(f"{i:<10} | {source:<10} | {name}")
        print("="*60 + "\n")

    if not hasattr(data, 'x') or data.x is None:
        raise ValueError("Error: No features provided.")

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
            att_name = str(pkl.removeprefix("graph_with_").removesuffix(".pkl"))
            
            # Normalize specific feature names (e.g., shortest_path_length)
            if 'shortest_path_length' in att_name:
                display_name = 'shortest_path_length'
            elif 'closeness' in att_name:
                display_name = 'closeness_centrality'
            else:
                display_name = att_name

            # 2. Check if this feature is requested in config
            if display_name in use_features:
                try:
                    # Load the attribute graph (contains node features)
                    attG = loadGraph(os.path.join(PATH, pkl))
                    attG = attG.subgraph(raw_nodes).copy()
                    
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

    if config.split_to_subgraphs:
        df = pd.read_csv(config.Feature_PATH+'node_mutation_with_BMR_v120525.csv',)
        df = df[['node_id', 'is_mut']]

        mut_mapping = dict(zip(df['node_id'], df['is_mut']))
        
        for node in finalG.nodes():
            val = mut_mapping.get(node, 0)
            finalG.nodes[node]['is_mut'] = val

    return finalG


def map_att_to_node(graph, attdf, use_cols=None, node_id_col='node_id', verbose=False):
    """Maps attributes from a DataFrame to graph nodes."""

    resultG = graph.copy()
    if use_cols:
        attdf = attdf[[node_id_col] + use_cols]

    attdf = attdf.set_index(node_id_col)
    node_list = list(graph.nodes)
    
    attr_dict = attdf.to_dict(orient='index')
    
    # Initialize default values for nodes missing in the dataframe
    for node in node_list:
        if node not in attr_dict:
            attr_dict[node] = {c: 0 for c in (use_cols or [])}
    
    # Custom logic: 'has_mutation' flag
    for node in node_list:
        val = attr_dict[node].get('total_mutations_count', 0)
        attr_dict[node]['has_mutation'] = 1 if val != 0 else 0
    
    # Filter only existing nodes and set attributes
    attr_dict = {n: attr_dict[n] for n in node_list if n in attr_dict}
    nx.set_node_attributes(resultG, attr_dict)
    
    if verbose:
        print("=========== Validation Example ===========")
        node_to_check = 'p30260_761_ser'
        if resultG.has_node(node_to_check):
            print(f"Attributes for node {node_to_check}:")
            pprint(resultG.nodes[node_to_check])
        else:
            print(f"Node {node_to_check} not in the graph.")
            
        # Check an empty node (node in graph but not in DF)
        missing_nodes = list(set(node_list) - set(attdf.index))
        if missing_nodes:
            print("Node without Mutation Example:")
            pprint(graph.nodes[missing_nodes[0]])
        
    return resultG

def get_node_att_value(obj, att):
    """Helper to get attribute value from a Graph node or dict."""
    if obj is None:
        raise ValueError("Input object is None.")
    if isinstance(obj, nx.Graph):
        return [d.get(att, None) for n, d in obj.nodes(data=True)]
    elif isinstance(obj, dict):
        return obj.get(att, None)
    else:
        raise TypeError("Input must be a networkx Graph or a node attribute dictionary.")

def get_edge_att_value(G, att):
    """Helper to get attribute value from Graph edges."""
    return [d.get(att, None) for u, v, d in G.edges(data=True)]

def get_sample(G):
    """Returns a sample node and edge from the graph."""
    sample_node = next(iter(G.nodes))
    sample_edge = next(iter(G.edges))
    return sample_node, sample_edge

def normalize_node_attribute(G, all_node_att, att_name_list, method='minmax'):

    graph = G.copy()
    Success_target = []

    for att_name in all_node_att:

        if att_name not in att_name_list:
            nodes = list(graph.nodes())
            for i, n in enumerate(nodes):
                graph.nodes[n][att_name] = G.nodes[n][att_name]
            continue

        try:
            nodes = list(graph.nodes())
            values = np.array([graph.nodes[n].get(att_name, 0) for n in nodes], dtype=float)

            if method == 'minmax':
                min_val = np.min(values)
                max_val = np.max(values)
                if max_val - min_val == 0:
                    norm_values = values - min_val
                else:
                    norm_values = (values - min_val) / (max_val - min_val)
            
            elif method == 'zscore':
                mean_val = np.mean(values)
                std_val = np.std(values)
                if std_val == 0:
                    norm_values = values - mean_val
                else:
                    norm_values = (values - mean_val) / std_val

            for i, n in enumerate(nodes):
                graph.nodes[n][att_name] = float(norm_values[i])
            
            Success_target.append(att_name)
        
        except Exception as e:
            print(f"Failed to normalize attribute '{att_name} | using {method}': {str(e)}")
            continue
    print("============================"*2)
    print(f"Attributes '{Success_target}' normalized using {method}.")
    print("============================"*2)

    return graph