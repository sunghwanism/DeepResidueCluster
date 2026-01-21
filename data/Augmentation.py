import networkx as nx
from networkx.algorithms import isomorphism

def mutation_anchored_subgraphs(graph: nx.Graph, attribute: str, steps: int = 3) -> list:
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
    anchors = [n for n, data in graph.nodes(data=True) if data.get(attribute) == 1]
    unique_subgraphs = []
    
    nm = isomorphism.categorical_node_match(attribute, 0)

    for anchor in anchors:
        extended_dist = nx.single_source_shortest_path_length(graph, anchor, cutoff=steps)
        subgraph_nodes = list(extended_dist.keys())
        
        subg = graph.subgraph(subgraph_nodes).copy()
        
        is_duplicate = False
        for existing in unique_subgraphs:
            if nx.is_isomorphic(subg, existing, node_match=nm):
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_subgraphs.append(subg)
    
    print(f"Extracted {len(unique_subgraphs)} unique subgraphs.")

    return unique_subgraphs


def DataAugmentation(graph: nx.Graph, attribute: str, steps: int = 3, strategy: List[str] = None) -> list:
    """
    Augments the input graph based on the specified strategies.
    
    Args:
        graph (nx.Graph): The input graph.
        attribute (str): The node attribute to check (e.g., 'is_mut').
                         Nodes with data[attribute] == 1 are considered anchors.
        steps (int, optional): The number of steps (hops) from the anchor to include initially. Defaults to 3.
        strategy (List[str], optional): The strategies to apply. You can choose ['mut_anchored'] # TODO: add more strategies

    Returns:
        list[nx.Graph]: A list of unique (non-isomorphic) subgraphs.
    """
    
    if strategy is None:
        raise ValueError("No strategy specified.")
    
    for s in strategy:
        if s == "mut_anchored":
            yield mutation_anchored_subgraphs(graph, attribute, steps)
        else:
            raise ValueError(f"Unknown strategy: {s}")
