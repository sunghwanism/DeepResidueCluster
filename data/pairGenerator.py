import torch
import random
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from utils.table_utils import MAPPING_CONFIG


class GraphPairDataset(Dataset):
    """
    Custom Dataset for Graph-Level Contrastive Learning.
    
    Logic updated based on user request:
    1. Grouping:
       - '0-Graph': Contains node attribute 0.
       - '1-Graph': Contains node attribute 1.
    
    2. Pair Strategy:
       - Positive Pair (Label 1): (0, 0) OR (1, 1) -> Same semantic content.
       - Negative Pair (Label 0): (0, 1) OR (1, 0) -> Different semantic content.
    """

    def __init__(self, dataset, attribute_index=0, transform=None, pre_transform=None):
        """
        Args:
            dataset (list): List of PyG Data objects.
            attribute_index (int): The column index of the node feature to check.
        """
        super().__init__(None, transform, pre_transform)
        
        self.group_0 = []
        self.group_1 = []
        
        # Pre-process: Split dataset into two groups
        # Logic: 
        # Group 0 (Non-Mutated): All nodes have 0 mutations.
        # Group 1 (Mutated): At least one node has > 0 mutations.
        print(f"[Grouping] Using attribute index {attribute_index} for grouping...")
        for data in dataset:
            # Check if graph has any mutations (any node with attribute > 0)
            is_mutated = (data.x[:, attribute_index] > 0).any()
            
            if not is_mutated:
                self.group_0.append(data)
            else:
                self.group_1.append(data)

        # Validation: We need at least 2 graphs in each group to form same-class pairs
        if len(self.group_0) < 2 or len(self.group_1) < 2:
            raise ValueError(
                f"Insufficient data. Need at least 2 graphs in each group. "
                f"Current counts -> Group 0: {len(self.group_0)}, Group 1: {len(self.group_1)}"
            )

        print(f"[Dataset Ready] Non-Mutated Graphs {len(self.group_0)} || Mutated Graphs {len(self.group_1)}")

    def len(self):
        return len(self.group_0) + len(self.group_1)

    def get(self, idx):
        """
        Generates a pair.
        To ensure a balanced batch, we alternate between Positive and Negative pairs
        based on the index (even/odd).
        """
        
        if idx % 2 == 0:
            if random.random() > 0.5:
                graph_a, graph_b = self._sample_same_group(self.group_0)
            else:
                graph_a, graph_b = self._sample_same_group(self.group_1)
            
            label = torch.tensor([1.0], dtype=torch.float)
            
        else:
            graph_a = random.choice(self.group_0)
            graph_b = random.choice(self.group_1)
            
            label = torch.tensor([0.0], dtype=torch.float)

        if self.transform is not None:
            graph_a = self.transform(graph_a)
            graph_b = self.transform(graph_b)

        return graph_a, graph_b, label

    def _sample_same_group(self, group_list):
        """Helper to sample two DISTINCT graphs from the same group."""
        return random.sample(group_list, 2)

def collate_pairs(batch):
    """
    Collate function to batch pairs of graphs.
    """
    graph_a_list = [item[0] for item in batch]
    graph_b_list = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    # Create PyG Batches
    batch_a = Batch.from_data_list(graph_a_list)
    batch_b = Batch.from_data_list(graph_b_list)
    batch_labels = torch.stack(labels)

    return batch_a, batch_b, batch_labels

from torch_geometric.loader import NeighborLoader as PyGNeighborLoader

class NeighborSubgraphSampler:
    """
    Applies GraphSAGE-style Neighbor Sampling to a Data object.
    If sizes is a list (e.g. [25, 10]), it limits neighbors per layer.
    """
    def __init__(self, sizes):
        self.sizes = sizes

    def __call__(self, data):
        
        loader = PyGNeighborLoader(
            data,
            num_neighbors=self.sizes,
            batch_size=data.num_nodes, # Full batch
            shuffle=False,
            input_nodes=None, # All nodes are seeds
        )
        
        sampled_data = next(iter(loader))
        
        return sampled_data

def get_contrastive_loaders(train_data, val_data, test_data, config):
    """
    Creates DataLoaders for train, val, and test datasets.
    
    Args:
        train_data (list): List of PyG Data objects for training.
        val_data (list): List of PyG Data objects for validation.
        test_data (list): List of PyG Data objects for testing.
        config (object): Configuration object containing 'batch_size', 'num_workers', etc.
        
    Returns:
        train_loader, val_loader, test_loader
    """
    train_loader = get_contrastive_loader(train_data, config, shuffle=True)
    val_loader = get_contrastive_loader(val_data, config, shuffle=False)
    test_loader = get_contrastive_loader(test_data, config, shuffle=False)
    
    return train_loader, val_loader, test_loader


def find_attribute_index(config, target_name):
    """
    Finds the index of a feature in the node feature matrix (x) based on the config.
    Replicates the logic in utils/graph_utils.py (nx_to_pyg_data).
    """
    graph_features = getattr(config, 'graph_features', [])
    table_features = getattr(config, 'table_features', [])
    
    # 1. Check Graph Features
    if target_name in graph_features:
        return graph_features.index(target_name)
    
    # 2. Check Table Features (Numerical only, categorical are in x_cat)
    category_feat = [f for f in table_features if f in MAPPING_CONFIG]
    
    # Identify numerical table features (the logic from nx_to_pyg_data)
    split_features = [col for col in table_features if col not in category_feat]
    if 'ptms_mapped' in split_features:
        split_features.remove('ptms_mapped')
        
    # Search for variations
    for feat in split_features:
        feat_lower = feat.lower()
        target_lower = target_name.lower()
        
        # Exact match
        if feat == target_name:
            return len(graph_features) + split_features.index(feat)
            
        # Handle mutation vs mutations and count variations
        if 'total_mutation' in target_lower and 'total_mutation' in feat_lower and 'count' in feat_lower:
            return len(graph_features) + split_features.index(feat)
            
    return None

def get_contrastive_loader(dataset_list, config, shuffle=True):
    """
    Creates a single DataLoader from a list of data objects.
    Useful for inference or single-split loading.
    """
    if dataset_list is None or len(dataset_list) == 0:
        return None

    # Determine Sampling Transform
    transform = None
    num_sample_nodes = getattr(config, 'num_sample_nodes', None) # List of sizes for each layer
    
    if num_sample_nodes is not None:
        transform = NeighborSubgraphSampler(num_sample_nodes)
        
    # Automatically find attribute index for 'total_mutation_count' if possible
    # User can specify 'group_attribute' in config, defaults to 'total_mutation_count'
    target_attr = getattr(config, 'group_attribute', 'total_mutation_count')
    attr_idx = find_attribute_index(config, target_attr)
    
    if attr_idx is None:
        # Fallback to the manual index if name not found
        attr_idx = getattr(config, 'attribute_index', 0)
        print(f"[Warning] Attribute '{target_attr}' not found in features. Using fallback index: {attr_idx}")
    else:
        print(f"[Info] Found '{target_attr}' at index {attr_idx}")

    # Create Dataset with Transform
    dataset = GraphPairDataset(dataset_list, attribute_index=attr_idx, transform=transform)
    
    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle, # Shuffle pairs, not nodes
        num_workers=getattr(config, 'num_workers', 0),
        pin_memory=getattr(config, 'pin_memory', True),
        collate_fn=collate_pairs
    )
    
    return loader