import os
import time
from tqdm import tqdm

import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import torch
from torch_geometric.loader import DataLoader, NeighborLoader
from sklearn.model_selection import train_test_split

from utils.graph_utils import nx_to_pyg_data, loadGraph, merge_graph_attributes, filtered_only_attributes, get_sample
from utils.table_utils import make_bin_cols, scaling_and_fillnafeature
from data.Augmentation import mutation_anchored_subgraphs

def LoadDataset(config, only_test=False, clear_att_in_orginG=False):
    """
    Loads graph, splits into connected components, filters by size,
    converts to PyG Data, augments, and splits into train/val/test.
    """

    # 1. Load Graph Structure
    G = loadGraph(config.Graph_PATH)
    graph = merge_graph_attributes(G, config)

    print("Graph loaded successfully")
    print(graph)
    node, edge = get_sample(graph)
    print("Node Example", node)
    print("Edge Example", edge)
    del G, node, edge
    print("============================"*2)
    
    # 2. Load Table Features (Node Attributes)
    df = None 
    if config.table_features is not None:
        try:
            basic_node_df = pd.read_csv(os.path.join(config.Feature_PATH, 'node_features.csv'))
            am_node_df = pd.read_csv(os.path.join(config.Feature_PATH, 'node_features_with_am.csv'))
            bmr_df = pd.read_csv(os.path.join(config.Feature_PATH, 'node_mutation_with_BMR_v120525.csv'))
            bmr_df.drop(columns=['total_mutations_count', 'unique_mutation_types_count', 'unique_patients_count', 'uniprot_id'], inplace=True)

            df = pd.merge(basic_node_df, am_node_df, on='node_id', how='left')
            df = pd.merge(df, bmr_df, on='node_id', how='left')

            df = scaling_and_fillnafeature(df, config.table_features)

            if 'mut+res-bin' in config.table_features:
                df = make_bin_cols(df, 'mut+res-bin', bin_size=config.bin_size)

            df = df.set_index(config.node_col_name)
            if config.label_col is not None:
                cols = [config.label_col] + config.table_features
                df = df[cols]
            else:
                df = df[config.table_features]
                
            print("Features loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load CSV features. {e}")
            df = None
    print("============================"*2)

    # 3. Split Graph into Connected Components
    print("Splitting graph into connected components...")

    components = [
        graph.subgraph(c).copy() 
        for c in nx.connected_components(graph) 
        if config.min_cc_size <= len(c) <= config.max_cc_size
    ]
    print(f"Total Components found: {len(components)}")
    
    components = ProcessConnectedComponents(components, config)
    print(f"Components after filtering size: {len(components)}")
    print("============================"*2)

    AugmentedComponents = []

    if config.split_to_subgraphs and not config.PreProcessDATA:
        print("[DataAugmentation] Splitting components into small subgraphs...")
        start_time = time.time()
        
        for comp in components:
            aug_comp_list = mutation_anchored_subgraphs(
                comp, 
                'is_mut', 
                config.aug_subgraph_steps, 
                max_nodes=2000, 
                sample_ratio=0.3,   
                min_size=getattr(config, 'min_cc_size', 0), 
                max_size=getattr(config, 'max_cc_size', float('inf'))
            )
            final_comp_list = ProcessConnectedComponents(aug_comp_list, config)
            AugmentedComponents.extend(final_comp_list)

            AugmentedComponents.append(comp)

        components = AugmentedComponents.copy()

        plt.hist([comp.number_of_nodes() for comp in components], bins=50)
        plt.yscale('log')
        plt.ylabel('Number of Subgraphs (log)')
        plt.xlabel('Number of Nodes')
        plt.savefig("asset/subgraph_size.png")

        # Save augmented components to file
        if not config.PreProcessDATA:
            save_path = 
            with open(save_path, 'wb') as f:
                pickle.dump(components, f)
            print(f"Saved augmented components to {save_path}")

        del AugmentedComponents, aug_comp_list, final_comp_list
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total Subgraphs found: {len(components)}")
        print(f"Time taken for augmentation: {elapsed_time:.2f} seconds")
        print(f"Components after augmentation (incl. original CC): {len(components)}")
        print("============================"*2)

    # 4. Convert to PyG Data Objects
    if config.PreProcessDATA:
        with open(config.PreProcessDATA, 'rb') as f:
            components = pickle.load(f)
            
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
            add_constant_feature=False,
            config=config
        )
        data_list.append(pyg_obj)
    
    print("PyG Data conversion complete")
    print("============================"*2)

    # 5. Apply Data Augmentation
    data_list = DataAugmentation(data_list, config)
    print("Finish Data Augmentation")

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
    
    return train, test, val


def DataAugmentation(data_list, config):
    """
    Applies structural augmentations (e.g., constant features).
    """
    print("Start Data Augmentation")
    augmentation_log = []
    
    # 1. Add Constant Feature
    if getattr(config, 'add_const_to_node', False):
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
    
    # print(f"Filtering Components: Min {min_size} <= Nodes <= Max {max_size}")

    filtered_list = [
        g for g in components_list 
        if min_size <= g.number_of_nodes() <= max_size
    ]
    
    removed_count = len(components_list) - len(filtered_list)
    if removed_count > 0:
        print(f"Removed {removed_count} components outside size range.")

    return filtered_list


def getDataLoader(data_list, config, test=False):
    if config.num_sample_nodes is None:
        return DataLoader(data_list,
                          batch_size=config.batch_size,
                          pin_memory=True,
                          num_workers=config.num_workers,
                          shuffle=not test)
    else:
        from torch_geometric.data import Batch
        merged_data = Batch.from_data_list(data_list)
        
        return NeighborLoader(
            merged_data,
            num_neighbors=config.num_sample_nodes,
            batch_size=config.batch_size,
            shuffle=not test,
            num_workers=config.num_workers,
            pin_memory=True
        )