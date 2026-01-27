import networkx as nx
from networkx.algorithms import isomorphism

import random

def mutation_anchored_subgraphs(graph: nx.Graph, attribute: str, steps: int = 3, max_nodes: int = 2000, sample_ratio: float = 0.3, min_size: int = 0, max_size: int = float('inf')) -> list:
    """
    Extracts node-anchored subgraphs based on specific criteria.
    
    For each node where graph.nodes[n][attribute] == 1:
        1. Extract nodes within 'steps' distance from the anchor.
        2. Create the induced subgraph.
        3. Return a list of non-isomorphic subgraphs.

    Args:
        graph (nx.Graph): The input graph.
        attribute (str): The node attribute to check (e.g., 'is_mut').
                         Nodes with data[attribute] == 1 are considered anchors.
        steps (int, optional): The number of steps (hops) from the anchor to include initially. Defaults to 3.

    Returns:
        list[nx.Graph]: A list of unique (non-isomorphic) subgraphs.
    """

    if graph.number_of_nodes() < 100:
        return [graph]

    anchors = [n for n, data in graph.nodes(data=True) if data.get(attribute) == 1]
    unique_subgraphs = []
    
    nm = isomorphism.categorical_node_match(attribute, 0)

    if sample_ratio < 1.0:
        num_samples = int(len(anchors) * sample_ratio)
        if num_samples > 2:
            anchors = random.sample(anchors, num_samples)
        else:
            anchors = anchors

    def get_nodes_from_bfs(start_node, depth, max_len=2000):
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

    for anchor in anchors:
        combined_nodes = get_nodes_from_bfs(anchor, steps, max_nodes)
        neighbors = list(graph.neighbors(anchor))
        for neighbor in neighbors:
            neighbor_nodes = get_nodes_from_bfs(neighbor, 2, max_nodes)
            combined_nodes.update(neighbor_nodes)

        subg = graph.subgraph(list(combined_nodes)).copy()
        
        if min_size <= subg.number_of_nodes() <= max_size:
            is_duplicate = False
            for existing in unique_subgraphs:
                if nx.is_isomorphic(subg, existing, node_match=nm):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_subgraphs.append(subg)
    
    return unique_subgraphs