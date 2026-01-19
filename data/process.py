import pickle
import pandas as pd
import networkx as nx

import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from utils.graph_utils import nx_to_pyg_data

def LoadDataset(config, only_test=False):
    """
    Loads graph, splits into connected components, filters by size,
    converts to PyG Data, augments, and splits into train/val/test.
    """

    # 1. Load Graph Structure
    with open(config.Graph_PATH, 'rb') as f:
        graph = pickle.load(f)

    print("Graph loaded successfully")
    print(f"Original Graph: {graph}") 
    print("============================ "*2)
    
    # 2. Load Table Features (Node Attributes)
    df = None 
    if config.table_features is not None:
        try:
            df = pd.read_csv(config.Feature_PATH)
            
            if config.label_col is not None:
                # Check if columns exist
                cols = [config.label_col] + config.table_features
                df = df[cols]
            else:
                df = df[config.table_features]
                
            print("Features loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load CSV features. {e}")
            df = None
    print("============================ "*2)

    # 3. Split Graph into Connected Components
    print("Splitting graph into connected components...")
    
    components = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    print(f"Total Components found: {len(components)}")
    
    components = ProcessConnectedComponents(components, config)
    print(f"Components after filtering size: {len(components)}")
    print("============================ "*2)

    # 4. Convert to PyG Data Objects
    data_list = []
    print("Converting to PyG Data objects...")
    for comp in components:
        pyg_obj = nx_to_pyg_data(
            comp, 
            node_features_df=df, 
            label_col=config.label_col, 
            graph_features=config.graph_features, 
            table_features=config.table_features, 
            use_edge_weight=config.use_edge_weight,
            add_constant_feature=False 
        )
        data_list.append(pyg_obj)
    
    print("PyG Data conversion complete")
    print("============================ "*2)

    # 5. Apply Data Augmentation
    data_list = DataAugmentation(data_list, config)
    print("Finish Data Augmentation")
    print("============================ "*2)

    # 6. Split Dataset (Train / Val / Test)
    if only_test:
        return data_list

    # Check validity of ratios
    if sum(config.ratio_of_data) != 1.0:
        raise ValueError("Data split ratios do not sum to 1.0")

    # Split: Train+Val (remaining) / Test
    train_val, test = train_test_split(
        data_list, 
        test_size=config.ratio_of_data[2], 
        random_state=config.SEED
    )

    # Calculate relative validation size
    # Val ratio relative to (Train + Val)
    denom = config.ratio_of_data[0] + config.ratio_of_data[1]        
    relative_val_size = config.ratio_of_data[1] / denom
    
    train, val = train_test_split(
        train_val, 
        test_size=relative_val_size, 
        random_state=config.SEED
    )

    print(f"Dataset Split -> Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    return train, test, val


def DataAugmentation(data_list, config):
    """
    Applies structural augmentations (e.g., constant features).
    """
    print("Start Data Augmentation")
    augmentation_log = []
    
    # 1. Add Constant Feature
    if getattr(config, 'add_constant_feature', False):
        for PyGData in data_list:
            num_nodes = PyGData.num_nodes # PyG object uses .num_nodes
            constant_x = torch.ones((num_nodes, 1), dtype=torch.float)
            
            if hasattr(PyGData, 'x') and PyGData.x is not None:
                PyGData.x = torch.cat([PyGData.x, constant_x], dim=1)
            else:
                PyGData.x = constant_x
                
        augmentation_log.append("Constant Feature")

    # 2. Future Augmentations (e.g., Virtual Nodes, Edge Perturbation)
    # if config.use_virtual_node:
    #     ...

    print(f"Applied Augmentations: {augmentation_log}")
    return data_list


def ProcessConnectedComponents(components_list, config):
    """
    Filters Connected Components (NetworkX Graphs) based on node count.
    
    Args:
        components_list (list): List of NetworkX Graph objects.
        config: Config object with min_cc_size and max_cc_size.
    """
    
    min_size = getattr(config, 'min_cc_size', 0)
    max_size = getattr(config, 'max_cc_size', float('inf'))
    
    print(f"Filtering Components: Min {min_size} <= Nodes <= Max {max_size}")

    filtered_list = [
        g for g in components_list 
        if min_size <= g.number_of_nodes() <= max_size
    ]
    
    removed_count = len(components_list) - len(filtered_list)
    if removed_count > 0:
        print(f"Removed {removed_count} components outside size range.")

    return filtered_list


def getDataLoader(data_list, config, test=False):
    dataloader = DataLoader(data_list,
                            batch_size=config.batch_size,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            shuffle=True if not test else False)
    return dataloader