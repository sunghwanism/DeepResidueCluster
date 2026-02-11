
import os
import pickle
import torch
import pandas as pd
import networkx as nx
from typing import Optional, Tuple, List, Any, Union
from torch_geometric.data import DataLoader, Data, Batch
from torch_geometric.loader import NeighborLoader
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger
from src.utils.graph_ops import (
    load_graph, 
    merge_graph_attributes, 
    normalize_node_attribute, 
    nx_to_pyg_data
)
from src.data.transforms import MutationSubgraphAugment, AddConstantFeature
from src.data.datasets import GraphPairDataset, NeighborSubgraphSampler, find_attribute_index # Assuming get_contrastive_loader logic needs to be inside DataModule or imported
# Note: get_contrastive_loader was in pairGenerator.py but I didn't fully port it to datasets.py in previous step.
# I should probably implement the loader creation logic inside DataModule or add it to datasets.py.
# For now, I will implement loader creation inside DataModule for better cohesion.

logger = get_logger(__name__)

class DeepResidueDataModule:
    """
    DataModule to handle all data loading, processing, splitting, and loader creation.
    """
    def __init__(self, config: Any):
        self.config = config
        self.train_data: List[Data] = []
        self.val_data: List[Data] = []
        self.test_data: List[Data] = []
        
    def setup(self, stage: Optional[str] = None):
        """
        Loads data, processes it, and splits it.
        """
        logger.info("Setting up DataModule...")
        
        # 1. Load and Preprocess Graph
        graph = self._load_and_process_graph()
        
        # 2. Load Node Features (Table)
        df, final_table_features = self._load_table_features()
        
        # 3. Split into Connected Components
        components = self._split_to_components(graph)
        
        # 4. Split Dataset (Train/Val/Test)
        train_comps, val_comps, test_comps = self._split_dataset(components)
        
        # 5. Data Augmentation (Train only)
        # Note: Original code augmented NetworkX graphs BEFORE converting to PyG
        # But applied Feature Augmentation AFTER converting to PyG.
        
        if self.config.use_aug and getattr(self.config, 'split_to_subgraphs', False):
             train_comps = self._augment_graph_topology(train_comps)

        # 6. Convert to PyG Data
        logger.info("Converting to PyG Data objects...")
        self.train_data = self._convert_to_pyg(train_comps, "Train", df, final_table_features)
        self.val_data = self._convert_to_pyg(val_comps, "Val", df, final_table_features)
        self.test_data = self._convert_to_pyg(test_comps, "Test", df, final_table_features)
        
        # 7. Feature Augmentation (e.g. Constant Feature)
        self._apply_feature_augmentation()
        
        logger.info(f"Data Setup Complete. Train: {len(self.train_data)}, Val: {len(self.val_data)}, Test: {len(self.test_data)}")

    def train_dataloader(self) -> DataLoader:
        if self.config.model == 'DGI':
             return self._get_standard_loader(self.train_data, shuffle=True)
        elif self.config.model == 'pchk':
             # Contrastive Loader
             return self._get_contrastive_loader(self.train_data, shuffle=True)
        else:
             return self._get_standard_loader(self.train_data, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.config.model == 'pchk':
             return self._get_contrastive_loader(self.val_data, shuffle=False)
        return self._get_standard_loader(self.val_data, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.config.model == 'pchk':
              return self._get_contrastive_loader(self.test_data, shuffle=False)
        return self._get_standard_loader(self.test_data, shuffle=False)

    def _load_and_process_graph(self) -> nx.Graph:
        path = self.config.Graph_PATH
        if not os.path.exists(path):
            logger.info("Graph path not found, attempting to merge attributes...")
            raise FileNotFoundError(f"Graph file not found at {path}")
        
        with open(path, 'rb') as f:
            mergedG = pickle.load(f)
            
        graph = normalize_node_attribute(
            mergedG, 
            self.config.graph_features, 
            self.config.node_att_norm_target, 
            method=self.config.node_att_norm_method
        )
        logger.info("Graph loaded and normalized.")
        return graph

    def _load_table_features(self) -> Tuple[Optional[pd.DataFrame], List[str]]:
        if self.config.table_features is None:
            return None, []

        try:
            # Construct paths
            feat_path = self.config.Feature_PATH
            # Hardcoded filenames from original logic
            basic_node_df = pd.read_csv(os.path.join(feat_path, 'node_features.csv'))
            am_node_df = pd.read_csv(os.path.join(feat_path, 'node_features_with_am_v02062026.csv'))
            bmr_df = pd.read_csv(os.path.join(feat_path, 'node_mutation_with_BMR_v120525.csv'))
            
            # Merge
            cols_to_drop = ['total_mutations_count', 'unique_mutation_types_count', 'unique_patients_count', 'uniprot_id']
            bmr_df = bmr_df.drop(columns=cols_to_drop)
            
            df = pd.merge(basic_node_df, am_node_df, on='node_id', how='left')
            df = pd.merge(df, bmr_df, on='node_id', how='left')
            
            from src.utils.sub_ops import make_bin_cols, scaling_and_fillna_feature, process_ptms
            
            # Pre-processing
            df = scaling_and_fillna_feature(df, self.config.table_features)
            
            if 'mut+res-bin' in self.config.table_features:
                 df = make_bin_cols(df, 'mut+res-bin', bin_size=self.config.bin_size)
            
            final_table_features = list(self.config.table_features) # Copy
            
            if 'ptms_mapped' in self.config.table_features:
                logger.info("Processing PTMs features globally...")
                df, ptm_cols = process_ptms(df)
                final_table_features = [f for f in final_table_features if f != 'ptms_mapped']
                final_table_features.extend(ptm_cols)
            
            df = df.set_index(self.config.node_col_name)
            
            if self.config.label_col is not None:
                cols = [self.config.label_col] + list(df.columns) # Keep all for now or filter?
                # Original code filtered: cols = [label] + features
                # But we need to keep features available specifically
                pass 
                
            logger.info("Table Features loaded successfully")
            return df, final_table_features
            
        except Exception as e:
            logger.warning(f"Failed to load CSV features: {e}")
            return None, self.config.table_features

    def _split_to_components(self, graph: nx.Graph) -> List[nx.Graph]:
        min_size = getattr(self.config, 'min_cc_size', 0)
        max_size = getattr(self.config, 'max_cc_size', float('inf'))
        
        components = [
            graph.subgraph(c).copy() 
            for c in nx.connected_components(graph) 
            if min_size <= len(c) <= max_size
        ]
        logger.info(f"Total Components found and filtered: {len(components)}")
        return components

    def _split_dataset(self, components: List[nx.Graph]) -> Tuple[List[nx.Graph], List[nx.Graph], List[nx.Graph]]:
        # Sort components
        components.sort(key=lambda x: x.number_of_nodes(), reverse=True)
        
        # Check for pre-saved splits
        train_path = f"{self.config.project_name}_train.pkl"
        if os.path.exists(train_path):
             logger.info("Loading existing splits...")
             def load(name):
                 with open(f"{self.config.project_name}_{name}.pkl", 'rb') as f:
                     return pickle.load(f)
             return load('train'), load('val'), load('test')

        # Create Splits
        mandatory_train = components[:3] # First 3 components are mandatory for training
        remaining = components[3:]
        
        train_val, test = train_test_split(remaining, test_size=self.config.ratio_of_data[2], random_state=self.config.SEED)
        
        denom = self.config.ratio_of_data[0] + self.config.ratio_of_data[1]
        relative_val = self.config.ratio_of_data[1] / denom
        
        train, val = train_test_split(train_val, test_size=relative_val, random_state=self.config.SEED)
        
        train = mandatory_train + train
        
        # Save mechanism should be here
        return train, val, test

    def _augment_graph_topology(self, components: List[nx.Graph]) -> List[nx.Graph]:
        logger.info("[DataAugmentation] Augmenting Training Set...")
        augmentor = MutationSubgraphAugment(
            attribute='is_mut',
            steps=self.config.aug_subgraph_steps,
            sample_ratio=0.1
        )
        
        augmented_comps = []
        for comp in components:
            augmented_comps.append(comp)
            aug_list = augmentor(comp)
            augmented_comps.extend(aug_list)
            
        logger.info(f"Augmentation result: {len(components)} -> {len(augmented_comps)}")
        return augmented_comps

    def _convert_to_pyg(self, comp_list, split_name, df, final_features) -> List[Data]:
        pyg_list = []
        for i, comp in enumerate(comp_list):
            pyg_obj = nx_to_pyg_data(
                comp,
                node_features_df=df,
                label_col=self.config.label_col,
                graph_features=self.config.graph_features,
                table_features=final_features,
                use_edge_weight=self.config.use_edge_weight,
                add_constant_feature=False, # We do this in step 7 separately if needed or here? 
                # Original did it separately in FeatureAugmentation.
                config=self.config
            )
            pyg_list.append(pyg_obj)
        return pyg_list

    def _apply_feature_augmentation(self):
         if getattr(self.config, 'add_const_to_node', False):
             aug = AddConstantFeature()
             for data in self.train_data + self.val_data + self.test_data:
                 aug(data)
             logger.info("Applied Constant Feature Augmentation")

    def _get_standard_loader(self, data_list, shuffle):
        if self.config.num_sample_nodes is None:
            return DataLoader(
                data_list, 
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                num_workers=self.config.num_workers,
                pin_memory=True
            )
        else:
            # Neighbor Loader
            batch = Batch.from_data_list(data_list)
            return NeighborLoader(
                batch,
                num_neighbors=self.config.num_sample_nodes,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                num_workers=self.config.num_workers
            )

    def _get_contrastive_loader(self, data_list, shuffle):
        # Using logic from custom GraphPairDataset
        
        transform = None
        if getattr(self.config, 'num_sample_nodes', None) is not None:
             transform = NeighborSubgraphSampler(self.config.num_sample_nodes)

        # Find attribute index
        target_attr = getattr(self.config, 'group_attribute', 'total_mutation_count')
        attr_idx = find_attribute_index(self.config, target_attr)
        if attr_idx is None:
             attr_idx = 0
             logger.warning(f"Attribute {target_attr} not found, using index 0")

        dataset = GraphPairDataset(
            data_list, 
            attribute_index=attr_idx, 
            transform=transform
        )
        
        from src.data.datasets import collate_pairs
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            collate_fn=collate_pairs
        )
