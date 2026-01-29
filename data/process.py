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

from utils.graph_utils import nx_to_pyg_data, loadGraph, merge_graph_attributes, filtered_only_attributes, get_sample, normalize_node_attribute
from utils.table_utils import make_bin_cols, scaling_and_fillnafeature
from data.Augmentation import mutation_anchored_subgraphs

def LoadDataset(config, only_test=False, clear_att_in_orginG=False):
    """
    Loads graph, splits into connected components, filters by size,
    converts to PyG Data, augments, and splits into train/val/test.
    """

    # 1. Load Graph Structure
    G = loadGraph(config.Graph_PATH)
    mergedG = merge_graph_attributes(G, config)

    # Normalize node attributes
    graph = normalize_node_attribute(mergedG, config.graph_features, config.node_att_norm_target, method=config.node_att_norm_method)

    print("Graph loaded successfully")
    print(graph)
    node, edge = get_sample(graph)
    print("Node Example", node)
    print("Edge Example", edge)
    del G, node, edge, mergedG
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

    # 4. Split Dataset (Train / Val / Test) BEFORE Augmentation
    if sum(config.ratio_of_data) != 1.0:
        raise ValueError("Data split ratios do not sum to 1.0")

    # Sort components by number of nodes (descending)
    components.sort(key=lambda x: x.number_of_nodes(), reverse=True)

    if not only_test:
        if not os.path.exists(f"{config.project_name}_train.pkl"):
            # Force top 5 into Train
            mandatory_train = components[:5]
            remaining_components = components[5:]
            
            print(f"Top 5 components (Nodes: {[c.number_of_nodes() for c in mandatory_train]}) forced into Train.")

            train_val_comps, test_comps = train_test_split(
                remaining_components, 
                test_size=config.ratio_of_data[2], 
                random_state=config.SEED
            )
            
            denom = config.ratio_of_data[0] + config.ratio_of_data[1]        
            relative_val_size = config.ratio_of_data[1] / denom
            
            train_comps, val_comps = train_test_split(
                train_val_comps, 
                test_size=relative_val_size, 
                random_state=config.SEED
            )
            
            # Add mandatory components to train
            train_comps = mandatory_train + train_comps
            
            print(f"Split counts - Train: {len(train_comps)} (incl {len(mandatory_train)} mandatory), Val: {len(val_comps)}, Test: {len(test_comps)}")

            # Save Pre-Augmentation Splits
            for name, data in [('train', train_comps), ('val', val_comps), ('test', test_comps)]:
                save_path = f"{config.project_name}_{name}.pkl"
                with open(save_path, 'wb') as f:
                    pickle.dump(data, f)
                print(f"Saved {name} split (pre-aug) to {save_path}")

        else:
            # Load Pre-Augmentation Splits
            train_comps = None
            val_comps = None
            test_comps = None

            for name in ['train', 'val', 'test']:
                save_path = f"{config.project_name}_{name}.pkl"
                with open(save_path, 'rb') as f:
                    if name == 'train':
                        if config.use_aug and os.path.exists(f'{config.project_name}_train_aug.pkl'):
                            save_path = f'{config.project_name}_train_aug.pkl'
                            with open(save_path, 'rb') as f:
                                train_comps = pickle.load(f)
                        else:
                            train_comps = pickle.load(f)
                    elif name == 'val':
                        val_comps = pickle.load(f)
                    elif name == 'test':
                        test_comps = pickle.load(f)

                if config.use_aug:
                    print(f"Loaded {name} split (augmented) from {save_path}")
                else:
                    print(f"Loaded {name} split (pre-augmented) from {save_path}")

        # 5. Data Augmentation (Only on Train)
        if config.use_aug and config.split_to_subgraphs:
            if os.path.exists(f'{config.project_name}_train_aug.pkl'):
                pass

            else:
                print("[DataAugmentation] Augmenting Training Set...")
                start_time = time.time()
                
                augmented_train_comps = []
                for comp in train_comps:
                    # Add Original
                    augmented_train_comps.append(comp)
                    
                    # Augment
                    aug_comp_list = mutation_anchored_subgraphs(
                        comp, 
                        'is_mut',
                        config.aug_subgraph_steps, 
                        sample_ratio=0.1, # This might be parameterizable
                        min_size=4, 
                        max_size=2000
                    )
                    final_aug_list = ProcessConnectedComponents(aug_comp_list, config)
                    augmented_train_comps.extend(final_aug_list)
                    
                end_time = time.time()
                print(f"Augmentation time: {end_time - start_time:.2f}s")
                print(f"Train components: {len(train_comps)} -> {len(augmented_train_comps)}")
                train_comps = augmented_train_comps
                
                # Save Post-Augmentation Splits (Val/Test are same as pre-aug but saved for consistency)
                save_path = f"{config.project_name}_train_aug.pkl"
                with open(save_path, 'wb') as f:
                    pickle.dump(train_comps, f)
                print(f"Saved Train split (post-aug/final) to {save_path}")
        
    elif only_test:
        print("Converting to PyG Data objects...")
        test_data = convert_to_pyg(components, "Test", df, config)
        print("Applying Feature Augmentation...")
        test_data = FeatureAugmentation(test_data, config)
        return test_data

    # 6. Convert to PyG Data Objects
    print("Converting to PyG Data objects...")

    train_data = convert_to_pyg(train_comps, "Train", df, config)
    print(f"Finish Train Data Conversion to PyG")
    val_data = convert_to_pyg(val_comps, "Val", df, config)
    print(f"Finish Val Data Conversion to PyG")
    test_data = convert_to_pyg(test_comps, "Test", df, config)
    print(f"Finish Test Data Conversion to PyG")

    # 7. Apply Feature Augmentation
    print("Applying Feature Augmentation...")
    train_data = FeatureAugmentation(train_data, config)
    val_data = FeatureAugmentation(val_data, config)
    test_data = FeatureAugmentation(test_data, config)
    
    return train_data, test_data, val_data


def FeatureAugmentation(data_list, config):
    """
    Applies feature augmentations (e.g., constant features).
    """
    print("Start Feature Augmentation")
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

    print(f"Applied Feature Augmentations: {augmentation_log}")
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

def convert_to_pyg(comp_list, split_name, df, config):
    pyg_list = []
    for comp in comp_list:
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
        pyg_list.append(pyg_obj)
    return pyg_list