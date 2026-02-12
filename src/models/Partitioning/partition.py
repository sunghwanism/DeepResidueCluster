import os
import networkx as nx
import numpy as np

import pandas as pd

def remove_all_repulsive_edge(G):
    """
    Remove all repulsive edges from the graph.
    
    Parameters:
    G (networkx.Graph): The input graph with 'sign' attribute on edges.
    
    Returns:
    networkx.Graph: The graph with repulsive edges removed.
    """
    H = G.copy()
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > 0]
    H.remove_edges_from(edges_to_remove)
    S = [H.subgraph(c).copy() for c in nx.connected_components(H)]

    return S

def remove_repulsive_with_zero_edge(G):
    """
    Remove repulsive edges that are connected to nodes with zero degree.
    
    Parameters:
    G (networkx.Graph): The input graph with 'sign' attribute on edges.
    
    Returns:
    networkx.Graph: The graph with specific repulsive edges removed.
    """
    H = G.copy()
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] >=0]
    H.remove_edges_from(edges_to_remove)
    S = [H.subgraph(c).copy() for c in nx.connected_components(H)]

    return S

def remove_repulsive_with_threshold(G, th=25):
    """
    Remove repulsive edges that are connected to nodes with degree less than or equal to a threshold.
    
    Parameters:
    G (networkx.Graph): The input graph with 'sign' attribute on edges.
    threshold (int): strictly attractive set | keep only edges with sufficient strength 
    (|E_total_rank|) >= threshold(+)
    
    Returns:
    networkx.Graph: The graph with specific repulsive edges removed.
    """
    H = G.copy()
    all_abs_edge_values = [abs(d['weight']) for u,v,d in H.edges(data=True)]
    threshold = np.percentile(all_abs_edge_values, int(th))
    print(f"Percentile-based threshold: {threshold}")
    
    edges_to_remove_with_threshold = [(u,v) for u,v,d in H.edges(data=True) if abs(d['weight']) <= threshold]
    edges_to_remove_with_threshold.extend([(u,v) for u,v,d in H.edges(data=True) if d['weight'] >=0])
    
    H.remove_edges_from(edges_to_remove_with_threshold)
    S = [H.subgraph(c).copy() for c in nx.connected_components(H)]

    return S

def change_cluster_from_threshold(graph_list: List[nx.Graph], min_nodes: int = 2) -> Dict[int, List[str]]:
    """Helper to convert graph list to cluster dict."""
    cluster_dict = {}
    for idx, nxG in enumerate(graph_list):
        nodes = list(nxG.nodes())
        if len(nodes) >= min_nodes:
            cluster_dict[idx] = nodes
    return cluster_dict
