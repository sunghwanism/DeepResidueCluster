
import os
import random
import glob
from typing import List, Tuple, Optional, Any
from torch_geometric.data import Dataset, DataLoader

from src.utils.sub_ops import MAPPING_CONFIG
from src.data.transforms import GraphTransform

class GraphPairDataset(Dataset):
    """
    Custom Dataset for Graph-Level Contrastive Learning.
    
    Logic:
    1. Grouping:
       - '0-Graph': Contains node attribute 0.
       - '1-Graph': Contains node attribute 1.
    
    2. Pair Strategy:
       - Positive Pair (Label 1): (0, 0) OR (1, 1) -> Same semantic content.
       - Negative Pair (Label 0): (0, 1) OR (1, 0) -> Different semantic content.
    """

    def __init__(self, dataset: List["Data"], attribute_index: int = 0, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        
        self.group_0 = []
        self.group_1 = []
        self.attribute_index = attribute_index
        
        # Pre-process: Split dataset into two groups
        print(f"[Grouping] Using attribute index {attribute_index} for grouping...")
        for data in dataset:
            # Check if graph has any mutations (any node with attribute > 0)
            # Assuming attribute_index points to a column in x
            if data.x.shape[1] > attribute_index:
                 is_mutated = (data.x[:, attribute_index] > 0).any()
            else:
                 # Fallback/Edge case
                 is_mutated = False

            if not is_mutated:
                self.group_0.append(data)
            else:
                self.group_1.append(data)

        if len(self.group_0) < 2 or len(self.group_1) < 2:
            print(
                f"[WARNING] Insufficient data. Need at least 2 graphs in each group. "
                f"Counts -> Group 0: {len(self.group_0)}, Group 1: {len(self.group_1)}"
            )

        print(f"[Dataset Ready] Non-Mutated {len(self.group_0)} || Mutated {len(self.group_1)}")

    def len(self):
        return len(self.group_0) + len(self.group_1)

    def get(self, idx):
        if idx % 2 == 0:
            if random.random() > 0.5:
                graph_a, graph_b = self._sample_same_group(self.group_0)
            else:
                graph_a, graph_b = self._sample_same_group(self.group_1)
            
            import torch
            label = torch.tensor([1.0], dtype=torch.float)
            
        else:
            graph_a = random.choice(self.group_0)
            graph_b = random.choice(self.group_1)
            
            import torch
            label = torch.tensor([0.0], dtype=torch.float)

        if self.transform is not None:
            # Note: Transform logic for pair needs care. 
            # If transform expects single data, we call it twice.
            pass 

        return graph_a, graph_b, label

    def _sample_same_group(self, group_list):
        if len(group_list) < 2:
             # Fallback if group is too small (should be caught in init)
             return group_list[0], group_list[0]
        return random.sample(group_list, 2)

def collate_pairs(batch):
    import torch
    from torch_geometric.data import Batch
    graph_a_list = [item[0] for item in batch]
    graph_b_list = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    batch_a = Batch.from_data_list(graph_a_list)
    batch_b = Batch.from_data_list(graph_b_list)
    batch_labels = torch.stack(labels)

    return batch_a, batch_b, batch_labels

class NeighborSubgraphSampler:
    """
    Applies GraphSAGE-style Neighbor Sampling.
    """
    def __init__(self, sizes: List[int]):
        self.sizes = sizes

    def __call__(self, data: "Data") -> "Data":
        from torch_geometric.loader import NeighborLoader as PyGNeighborLoader
        loader = PyGNeighborLoader(
            data,
            num_neighbors=self.sizes,
            batch_size=data.num_nodes,
            shuffle=False,
            input_nodes=None,
        )
        sampled_data = next(iter(loader))
        return sampled_data

def find_attribute_index(config: Any, target_name: str) -> Optional[int]:
    """Finds the index of a feature in the node feature matrix (x)."""
    graph_features = getattr(config, 'graph_features', [])
    table_features = getattr(config, 'table_features', [])
    
    if target_name in graph_features:
        return graph_features.index(target_name)
    
    category_feat = [f for f in table_features if f in MAPPING_CONFIG]
    split_features = [col for col in table_features if col not in category_feat]
    
    if 'ptms_mapped' in split_features:
        split_features.remove('ptms_mapped')
        
    for feat in split_features:
        feat_lower = feat.lower()
        target_lower = target_name.lower()
        
        if feat == target_name:
            return len(graph_features) + split_features.index(feat)
            
        if 'total_mutation' in target_lower and 'total_mutation' in feat_lower and 'count' in feat_lower:
            return len(graph_features) + split_features.index(feat)
            
    return None
