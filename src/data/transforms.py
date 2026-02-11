
import torch
import networkx as nx
import random
from typing import List, Optional, Union, Set, Any
from networkx.algorithms import isomorphism
from torch_geometric.data import Data
from src.utils.logger import get_logger

logger = get_logger(__name__)

class GraphTransform:
    """Base class for graph transformations."""
    def __call__(self, data: Any) -> Any:
        raise NotImplementedError

class AddConstantFeature(GraphTransform):
    """
    Adds a constant feature (1.0) to node features.
    """
    def __call__(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        constant_x = torch.ones((num_nodes, 1), dtype=torch.float)
        
        if hasattr(data, 'x') and data.x is not None:
            data.x = torch.cat([data.x, constant_x], dim=1)
        else:
            data.x = constant_x
            
        return data

class MutationSubgraphAugment:
    """
    Extracts node-anchored subgraphs based on specific mutation attributes.
    Refactored from `mutation_anchored_subgraphs`.
    """
    def __init__(self, 
                 attribute: str = 'is_mut', 
                 steps: int = 3, 
                 sample_ratio: float = 0.3,
                 min_size: int = 4,
                 max_size: int = 2000):
        self.attribute = attribute
        self.steps = steps
        self.sample_ratio = sample_ratio
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, graph: nx.Graph) -> List[nx.Graph]:
        if graph.number_of_nodes() < 100:
            return [graph]

        anchors = [n for n, data in graph.nodes(data=True) if data.get(self.attribute) == 1]
        unique_subgraphs = []
        
        nm = isomorphism.categorical_node_match(self.attribute, 0)

        if self.sample_ratio < 1.0:
            num_samples = int(len(anchors) * self.sample_ratio)
            if num_samples > 2:
                anchors = random.sample(anchors, num_samples)
        
        for anchor in anchors:
            combined_nodes = self._get_nodes_from_bfs(graph, anchor, self.steps, self.max_size)
            neighbors = list(graph.neighbors(anchor))
            for neighbor in neighbors:
                neighbor_nodes = self._get_nodes_from_bfs(graph, neighbor, 1, self.max_size)
                combined_nodes.update(neighbor_nodes)

            subg = graph.subgraph(list(combined_nodes)).copy()
            
            if self.min_size <= subg.number_of_nodes() <= self.max_size:
                if self._is_duplicate(subg, unique_subgraphs, nm):
                    continue
                
                if len(list(nx.connected_components(subg))) > 1:
                    continue
                    
                unique_subgraphs.append(subg)
        
        return unique_subgraphs

    def _get_nodes_from_bfs(self, graph: nx.Graph, start_node: Any, depth: int, max_len: int) -> Set[Any]:
        visited = {start_node}
        current_layer = {start_node}
        
        for _ in range(depth):
            next_layer = set()
            for node in current_layer:
                for neighbor in graph.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_layer.add(neighbor)
            
            if len(visited) > max_len:
                break
            
            current_layer = next_layer
            if not current_layer:
                break
        return visited

    def _is_duplicate(self, subg: nx.Graph, existing_list: List[nx.Graph], node_match) -> bool:
        for existing in existing_list:
            if nx.is_isomorphic(subg, existing, node_match=node_match):
                return True
        return False
