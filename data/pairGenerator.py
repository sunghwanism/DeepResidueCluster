import torch
import random
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch


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
        for data in dataset:
            # Check if any node has value 0 in the specified attribute column
            has_zero = (data.x[:, attribute_index] == 0).any()
            # Check if any node has value 1 in the specified attribute column
            has_one = (data.x[:, attribute_index] == 1).any()
            
            if has_zero:
                self.group_0.append(data)
            
            if has_one:
                self.group_1.append(data)

        # Validation: We need at least 2 graphs in each group to form same-class pairs
        if len(self.group_0) < 2 or len(self.group_1) < 2:
            raise ValueError(
                f"Insufficient data. Need at least 2 graphs in each group. "
                f"Current counts -> Group 0: {len(self.group_0)}, Group 1: {len(self.group_1)}"
            )

        print(f"Dataset Ready: {len(self.group_0)} '0-Graphs', {len(self.group_1)} '1-Graphs'.")

    def len(self):
        # Length is arbitrary in pair-based sampling, usually defined by the larger group size
        # or the total number of possible combinations. Here we approximate it to the total data size.
        return len(self.group_0) + len(self.group_1)

    def get(self, idx):
        """
        Generates a pair.
        To ensure a balanced batch, we alternate between Positive and Negative pairs
        based on the index (even/odd).
        """
        
        # Even index -> Generate Positive Pair (Same Class)
        if idx % 2 == 0:
            # Randomly decide whether to sample (0, 0) or (1, 1)
            # This ensures the model learns similarities for both classes.
            if random.random() > 0.5:
                # Case: (0, 0)
                graph_a, graph_b = self._sample_same_group(self.group_0)
            else:
                # Case: (1, 1)
                graph_a, graph_b = self._sample_same_group(self.group_1)
            
            # Label 1 for "Similar/Positive"
            label = torch.tensor([1.0], dtype=torch.float)
            
        # Odd index -> Generate Negative Pair (Diff Class)
        else:
            # Case: (0, 1) or (1, 0)
            graph_a = random.choice(self.group_0)
            graph_b = random.choice(self.group_1)
            
            # Label 0 for "Dissimilar/Negative"
            label = torch.tensor([0.0], dtype=torch.float)

        return graph_a, graph_b, label

    def _sample_same_group(self, group_list):
        """Helper to sample two DISTINCT graphs from the same group."""
        # random.sample ensures distinct indices if the list has unique objects
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


def get_contrastive_loader(dataset_list, config, shuffle=True):
    """
    Creates a single DataLoader from a list of data objects.
    Useful for inference or single-split loading.
    """
    if dataset_list is None or len(dataset_list) == 0:
        return None

    # Check for NeighborLoader request
    num_sample_nodes = getattr(config, 'num_sample_nodes', None)
    
    if num_sample_nodes is not None:
        # User requested NeighborLoader
        from torch_geometric.data import Batch
        from torch_geometric.loader import NeighborLoader
        
        merged_data = Batch.from_data_list(dataset_list)
        loader = NeighborLoader(
            merged_data,
            num_neighbors=num_sample_nodes,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=getattr(config, 'num_workers', 0),
            pin_memory=getattr(config, 'pin_memory', True)
        )
        return loader
        
    # Default attribute index to 0 if not present in config
    attr_idx = getattr(config, 'attribute_index', 0)
    
    # Create Dataset
    dataset = GraphPairDataset(dataset_list, attribute_index=attr_idx)
    
    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=getattr(config, 'num_workers', 0),
        pin_memory=getattr(config, 'pin_memory', True),
        collate_fn=collate_pairs
    )
    
    return loader