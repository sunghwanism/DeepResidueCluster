
import os
import pickle
import pandas as pd
import networkx as nx
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union

from src.utils.sub_ops import encode_features


def nx_to_pyg_data(
    G: nx.Graph,
    node_features_df: Optional[pd.DataFrame] = None,
    label_col: Optional[str] = None,
    graph_features: Optional[List[str]] = None,
    table_features: Optional[List[str]] = None,
    add_constant_feature: bool = True,
    use_edge_weight: bool = False,
    config: Any = None
) -> "Data":
    """
    Converts a NetworkX graph to a PyTorch Geometric Data object,
    merging features from both the graph attributes and an external DataFrame.
    """
    import torch
    from torch_geometric.data import Data
    from torch_geometric.utils import from_networkx
    
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
             # Identify missing nodes for better error reporting
            missing = [node for node in nodes if node not in node_features_df.index]
            print(f"[ERROR] Missing {len(missing)} nodes in features DF. First 5: {missing[:5]}")
            raise ValueError("Error: Not all nodes in the graph are present in the node_features_df index.")
        
        df_ordered = node_features_df.loc[nodes].copy()
        
        if label_col is not None:
             if label_col in df_ordered:
                labels = df_ordered[label_col].values
                data.y = torch.tensor(labels, dtype=torch.long)
        
        features_to_use = table_features if table_features is not None else [col for col in df_ordered.columns if col != label_col]
        
        if len(features_to_use) > 0:
            # Assuming config has mapping info or we pass a default path. 
            # Ideally config should be passed or mapping dir.
            mapping_dir = os.path.dirname(config.config_path) if config and hasattr(config, 'config_path') else 'config/mapping'
            # Fallback for mapping dir if config is not robust
            if not os.path.exists(mapping_dir):
                 mapping_dir = 'config/mapping'

            result_df, category_feat, numerical_feat = encode_features(df_ordered, features_to_use, mapping_dir)
            
            # Re-collect split features (expanding standard numeric + generated numeric)
            # Logic: features_to_use might contain 'ptms_mapped', which generates 'ptm_...'.
            # encode_features returns the final list of numerical cols including generated ones.
            
            # Record numerical table features
            feature_names.extend([f"[Table] {f}" for f in numerical_feat])
            
            if numerical_feat:
                table_x = torch.tensor(result_df[numerical_feat].values, dtype=torch.float)
                if hasattr(data, 'x') and data.x is not None:
                    data.x = torch.cat([data.x, table_x], dim=1)
                else:
                    data.x = table_x

            if category_feat:
                cat_tensor = torch.tensor(result_df[category_feat].values, dtype=torch.long)
                if hasattr(data, 'x_cat') and data.x_cat is not None:
                    data.x_cat = torch.cat([data.x_cat, cat_tensor], dim=1)
                else:
                    data.x_cat = cat_tensor

    if add_constant_feature:
        num_nodes = data.num_nodes
        constant_x = torch.ones((num_nodes, 1), dtype=torch.float)
        if hasattr(data, 'x') and data.x is not None:
            data.x = torch.cat([data.x, constant_x], dim=1)
        else:
            data.x = constant_x
        feature_names.append("[Extra] Constant Feature (1.0)")

    if not hasattr(data, 'x') or data.x is None:
        raise ValueError("Error: No features provided for the graph.")

    return data

def load_graph(path: str) -> nx.Graph:
    with open(path, 'rb') as f:
        G = pickle.load(f)
    return G

def filtered_only_attributes(G: nx.Graph, targets: Union[str, List[str]]) -> nx.Graph:
    resultG = G.copy()
    if isinstance(targets, str):
        targets = [targets]
    
    targets_set = set(targets)

    for n in resultG.nodes:
        # Create a new dict with only target attributes
        new_attrs = {k: v for k, v in resultG.nodes[n].items() if k in targets_set}
        resultG.nodes[n].clear()
        resultG.nodes[n].update(new_attrs)
            
    return resultG

def merge_graph_attributes(rawG: nx.Graph, config: Any) -> nx.Graph:
    """
    Merges node attributes from external pickle files into the main graph (rawG).
    """
    PATH = config.GraphAtt_PATH
    if not os.path.exists(PATH):
        print(f"[WARNING] Graph Attribute Path does not exist: {PATH}")
        return rawG

    files = os.listdir(PATH)
    use_features = config.graph_features
    filtered_features = []

    raw_nodes = set(rawG.nodes())
    num_raw_nodes = rawG.number_of_nodes()

    for pkl in files:
        if pkl.endswith('.pkl') and pkl.startswith('graph_with_'):
            att_name = str(pkl.removeprefix("graph_with_").removesuffix(".pkl"))
            
            # Normalize specific feature names
            if 'shortest_path_length' in att_name:
                display_name = 'shortest_path_length'
            elif 'closeness' in att_name:
                display_name = 'closeness_centrality'
            else:
                display_name = att_name

            if display_name in use_features:
                try:
                    attG = load_graph(os.path.join(PATH, pkl))
                    attG = attG.subgraph(raw_nodes).copy()
                except Exception as e:
                    print(f"[ERROR] Error loading {pkl}: {e}")
                    continue
                
                # Validation
                att_nodes = set(attG.nodes())
                if num_raw_nodes == attG.number_of_nodes() and att_nodes == raw_nodes:
                    # Merge attributes
                    for node, attrs in attG.nodes(data=True):
                        rawG.nodes[node].update(attrs)
                    
                    filtered_features.append(display_name)
                    print(f"Successfully merged feature: [{display_name}] from {pkl}")
                else:
                    print(f"[WARNING] Validation Failed for {pkl}: Node mismatch (Count or IDs)")

    # Ensure uniqueness
    filtered_features = list(set(filtered_features))
    finalG = filtered_only_attributes(rawG, targets=filtered_features)
    
    # Validation step
    for feat in filtered_features:
        missing_nodes = [n for n, d in finalG.nodes(data=True) if feat not in d]
        if missing_nodes:
            print(f"[ERROR] CRITICAL: Feature '{feat}' is missing in {len(missing_nodes)} nodes! Dropping.")
            # Logic to actually drop could be added here, but following original flow usually implies hard exit or warn

    if getattr(config, 'split_to_subgraphs', False):
         # This part seems specific to mutation logic, ensuring 'is_mut' exists
         # Ideally this dependency on a specific CSV path inside specific function is bad, 
         # but keeping it for functional parity for now.
         try:
            mut_df_path = os.path.join(config.Feature_PATH, 'node_mutation_with_BMR_v120525.csv')
            if os.path.exists(mut_df_path):
                df = pd.read_csv(mut_df_path)
                mut_mapping = dict(zip(df['node_id'], df['is_mut']))
                for node in finalG.nodes():
                    val = mut_mapping.get(node, 0)
                    finalG.nodes[node]['is_mut'] = val
         except Exception as e:
             print(f"[WARNING] Failed to load mutation mappin: {e}")

    return finalG


def normalize_node_attribute(G: nx.Graph, all_node_att: List[str], att_name_list: List[str], method: str = 'minmax') -> nx.Graph:

    graph = G.copy()
    success_target = []

    for att_name in all_node_att:
        if att_name not in att_name_list:
            # If not in target list, keep original values (already there since we copied)
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
            else:
                norm_values = values

            for i, n in enumerate(nodes):
                graph.nodes[n][att_name] = float(norm_values[i])
            
            success_target.append(att_name)
        
        except Exception as e:
            print(f"[ERROR] Failed to normalize attribute '{att_name} | using {method}': {str(e)}")
            continue

    print(f"Attributes '{success_target}' normalized using {method}.")
    return graph


def create_dgi_training_data(data: "Data"):
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
    import torch
    idx = torch.randperm(x.size(0))
    return x[idx]

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