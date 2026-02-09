import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import time
import pickle
import argparse
from pprint import pprint
from typing import Dict, Any

import pandas as pd
import networkx as nx



from utils_old.graph_utils import map_att_to_node, print_time
from utils_old.functions import clean_the_memory
from config import clustering_params

# Constants
USECOLS = ['unique_patients_count', 'total_mutations_count', 'unique_mutation_types_count']

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run clustering algorithms on a graph.")
    
    # Input/Output paths
    parser.add_argument('--node_path', type=str, required=True, help='Path to node attribute data (CSV)')
    parser.add_argument('--graph_path', type=str, required=True, help='Path to graph data (pickle)')
    parser.add_argument('--savepath', type=str, required=True, help='Directory to save results')
    parser.add_argument('--logpath', type=str, help='Path to save logs (optional)')
    
    # Execution parameters
    parser.add_argument('--measure', type=str, nargs='+', required=True, 
                        help='Clustering method(s) to run (e.g., MCL, DBSCAN, HDBSCAN). Use "all" for all.')
    
    return parser

def run_dbscan(G: nx.Graph, params: Dict[str, Any], config: argparse.Namespace) -> Dict[str, Any]:
    """Wrapper for DBSCAN execution, handling disconnected components."""
    from DBSCAN import dbscan_clustering_shortestpath
    
    clusters_dict = {}
    
    # DBSCAN implementation often requires connected components to calculate shortest paths
    if nx.number_connected_components(G) > 1:
        print(f"  -> Graph has {nx.number_connected_components(G)} connected components. Running on subgraphs...")
        for i, component_nodes in enumerate(nx.connected_components(G)):
            subgraph = G.subgraph(component_nodes).copy()
            sub_clusters, _ = dbscan_clustering_shortestpath(
                graph=subgraph,
                eps=params['eps'],
                min_samples=params['min_samples'],
                savePATH=config.savepath,
                config=config
            )
            # Prefix keys to ensure uniqueness
            for k, v in sub_clusters.items():
                clusters_dict[f"sub_{i}_{k}"] = v
    else:
        clusters_dict, _ = dbscan_clustering_shortestpath(
            graph=G,
            eps=params['eps'],
            min_samples=params['min_samples'],
            savePATH=config.savepath,
            config=config
        )
    return clusters_dict

def run_hdbscan(G: nx.Graph, params: Dict[str, Any], config: argparse.Namespace) -> Dict[str, Any]:
    """Wrapper for HDBSCAN execution, handling disconnected components."""
    from DBSCAN import hdbscan_clustering_shortestpath
    
    clusters_dict = {}
    
    if nx.number_connected_components(G) > 1:
        print(f"  -> Graph has {nx.number_connected_components(G)} connected components. Running on subgraphs...")
        for i, component_nodes in enumerate(nx.connected_components(G)):
            subgraph = G.subgraph(component_nodes).copy()
            sub_clusters, _ = hdbscan_clustering_shortestpath(
                graph=subgraph,
                min_cluster_size=params['min_cluster_size'],
                min_samples=params['min_samples'],
                cluster_selection_epsilon=params['cluster_selection_epsilon'],
                cluster_selection_method=params['cluster_selection_method'],
                allow_single_cluster=params['allow_single_cluster'],
                savePATH=config.savepath,
                config=config
            )
            for k, v in sub_clusters.items():
                clusters_dict[f"sub_{i}_{k}"] = v
    else:
        clusters_dict, _ = hdbscan_clustering_shortestpath(
            graph=G,
            min_cluster_size=params['min_cluster_size'],
            min_samples=params['min_samples'],
            cluster_selection_epsilon=params['cluster_selection_epsilon'],
            cluster_selection_method=params['cluster_selection_method'],
            allow_single_cluster=params['allow_single_cluster'],
            savePATH=config.savepath,
            config=config
        )
    return clusters_dict

def main(config):
    start_total = time.time()
    print("=" * 60)
    print("STARTING CLUSTERING PROCESS")
    pprint(vars(config))
    print("=" * 60)
    
    # 1. Load Data
    print(f"Loading graph from: {config.graph_path}")
    print(f"Loading attributes from: {config.node_path}")
    
    try:
        node_att_df = pd.read_csv(config.node_path)
        with open(config.graph_path, 'rb') as f:
            G = pickle.load(f)
            
        # Map attributes to graph nodes
        G = map_att_to_node(graph=G, attdf=node_att_df, 
                            use_cols=USECOLS, node_id_col='node_id',
                            verbose=False)
        
        print(f"[Graph Info] Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        print(f"[Graph Info] Connected Components: {nx.number_connected_components(G)}")
        print("=" * 60)

    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return

    # 2. Iterate through requested measures
    measures_to_run = config.measure
    if 'all' in measures_to_run:
        measures_to_run = list(clustering_params.keys())

    for measure in measures_to_run:
        if measure not in clustering_params:
            print(f"[Warning] Measure '{measure}' not found in config. Skipping...")
            continue
        
        print(f"Running Clustering: {measure}")
        measure_start = time.time()
        params = clustering_params[measure]
        pprint(params)
        
        # Result containers
        clusters_dict = {}
        extra_data = {} # To store M, cluster_info, edges, etc.

        try:
            if measure == 'DBSCAN':
                clusters_dict = run_dbscan(G, params, config)
                
            elif measure == 'HDBSCAN':
                clusters_dict = run_hdbscan(G, params, config)
            
            elif measure == 'MCL':
                from MCL import parallel_mcl_clustering
                # MCL naturally handles disconnected graphs via matrix operations
                clusters_dict, cluster_info, cluster_edges, M = parallel_mcl_clustering(
                    graph=G,
                    expansion_power=params['expansion_power'],
                    inflation_power=params['inflation_power'],
                    pruning_threshold=params['pruning_threshold'],
                    convergence_threshold=params['convergence_threshold'],
                    max_iter=params['max_iterations'],
                    use_weights=params['use_weights'],
                    weight_key=params['weight_key'],
                    min_cluster_size=params['min_cluster_size'],
                    singleton_clusters=params['singleton_clusters']
                )
                extra_data = {
                    'M_matrix': M,
                    'cluster_info': cluster_info,
                    'edge_info': cluster_edges
                }

            elif measure in ['DPClus', 'EAGLE']:
                print(f"[Info] {measure} is not yet implemented. Passing.")
                continue
            
            else:
                print(f"[Error] Unknown measure: {measure}")
                continue

            # Log results
            print(f"-> {measure} Completed. Found {len(clusters_dict)} clusters.")
            
            # Time calculation
            measure_end = time.time()
            H, m, s = time_calc(measure_start, measure_end)
            print(f"-> Execution Time: {H}h {m}m {s}s")
            
            # 3. Save Results
            formatted_time = time.strftime("%Y%m%d_%H%M%S")
            is_fragmented = nx.number_connected_components(G) > 1
            filename_suffix = "whole" if is_fragmented else "connected"
            
            save_filename = f'{measure}_clusters_{filename_suffix}_{formatted_time}.pkl'
            save_path = os.path.join(config.savepath, save_filename)
            
            # Construct save dictionary
            save_data = {
                'measure': measure,
                'params': params,
                'clusters': clusters_dict,
                'timestamp': formatted_time
            }
            save_data.update(extra_data) # Add M matrix, etc if exists
            
            os.makedirs(config.savepath, exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)
                
            print(f"[Saved] Results saved to: {save_path}")

        except ImportError as ie:
            print(f"[Error] Module import failed for {measure}. Check if the file exists. ({ie})")
        except Exception as e:
            print(f"[Error] An unexpected error occurred during {measure}: {e}")
            import traceback
            traceback.print_exc()

        print("=" * 60)

if __name__ == '__main__':
    parser = get_parser()
    run_config = parser.parse_args()
    main(run_config)