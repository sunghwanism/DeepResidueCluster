import networkx as nx

def eagle_clustering(G, density_min=0.2, pruning_min=0.5, seed_threshold=1.0):
    """
    Implements the core logic of the EAGLE (Extending the APPLAUSE Algorithm for Global Local Enhancement)
    algorithm. It uses seed selection based on node weighting and density-based expansion.

    Args:
        G (nx.Graph): The input network (networkx Graph object). Assumes an unweighted graph.
        density_min (float): Minimum cluster density threshold for final filtering. (e.g., 0.2)
        pruning_min (float): Minimum pruning threshold (connectivity ratio) for noise removal. (e.g., 0.5)
        seed_threshold (float): Multiplier for the seed's weight to determine the expansion threshold. (e.g., 1.0)
                                 Expansion condition: W(u) >= W(s) * seed_threshold

    Returns:
        list: A list where each element is a list of nodes belonging to a final, filtered cluster.
    """

    # 1. Calculate Node Weight W(v): W(v) = Density(v) * Degree(v)
    
    # Function to calculate subgraph density: Density(S) = |E_S| / (|S|*(|S|-1)/2)
    def calculate_subgraph_density(subgraph_nodes):
        if len(subgraph_nodes) < 2:
            return 0.0
        
        # Actual number of edges within the subgraph
        E_S = G.subgraph(subgraph_nodes).number_of_edges()
        
        # Maximum possible number of edges
        num_nodes = len(subgraph_nodes)
        Max_E = num_nodes * (num_nodes - 1) / 2
        
        return E_S / Max_E if Max_E > 0 else 0.0

    W_v = {} # Stores node weights W(v)
    all_nodes = list(G.nodes())

    for v in all_nodes:
        # Neighbor set S = Gamma(v)
        neighbors = list(G.neighbors(v))
        
        # Calculate Density(v): density of the subgraph induced by v and its neighbors
        subgraph_for_density = [v] + neighbors
        density_v = calculate_subgraph_density(subgraph_for_density)
        
        # Calculate Degree(v)
        degree_v = G.degree(v)
        
        # Calculate W(v)
        W_v[v] = density_v * degree_v

    # Store node weights as node attributes
    nx.set_node_attributes(G, W_v, 'weight')

    # 2. Iterative Cluster Extraction and Expansion
    available_nodes = set(G.nodes())
    clusters = []

    while available_nodes:
        # 2-1. Seed Selection: Choose the node s with the highest weight W(v) among available nodes
        seed_node = max(available_nodes, key=lambda n: W_v.get(n, -1), default=None)
        
        if seed_node is None:
            break

        current_cluster = {seed_node}
        available_nodes.remove(seed_node)
        
        # Expansion criterion threshold: W(s) * seed_threshold
        expansion_threshold = W_v[seed_node] * seed_threshold
        
        # 2-2. Cluster Expansion: Add neighbors u where W(u) >= threshold
        # Uses a breadth-first search (BFS) style queue for expansion
        
        queue = list(G.neighbors(seed_node)) 

        while queue:
            u = queue.pop(0)

            if u not in current_cluster and u in available_nodes:
                # Expansion Condition: W(u) >= expansion_threshold
                if W_v.get(u, 0) >= expansion_threshold:
                    current_cluster.add(u)
                    available_nodes.remove(u)
                    
                    # Add new neighbors to the queue
                    for neighbor in G.neighbors(u):
                        if neighbor not in current_cluster and neighbor in available_nodes and neighbor not in queue:
                            queue.append(neighbor)

        # Store the raw cluster before post-processing
        clusters.append(current_cluster)

    # 3. Cluster Filtering and Pruning (Post-processing)
    
    final_clusters = []
    for C in clusters:
        
        # 3-1. Density Filtering
        # Discard cluster if its initial density is below density_min
        if calculate_subgraph_density(C) < density_min:
            continue
        
        # 3-2. Pruning (Haircut) for noise removal
        C_pruned = set(C)
        pruned_happened = True
        
        while pruned_happened:
            pruned_happened = False
            nodes_to_remove = set()
            
            for u in C_pruned:
                # Edges between node u and other nodes within C_pruned
                edges_in_cluster = G.subgraph(C_pruned).degree(u)
                
                # Pruning Condition: (Edges in cluster / Total degree of u) >= pruning_min
                # Remove node u if its connectivity ratio to the cluster is too low
                if G.degree(u) > 0:
                    connectivity_ratio = edges_in_cluster / G.degree(u)
                else:
                    connectivity_ratio = 0.0

                if connectivity_ratio < pruning_min:
                    nodes_to_remove.add(u)
                    pruned_happened = True
            
            # Remove the identified nodes and repeat if any were removed
            C_pruned -= nodes_to_remove
            
            # Discard cluster if it becomes too small after pruning
            if len(C_pruned) < 2:
                break
        
        # Only accept the pruned cluster if it still has at least 2 nodes and maintains density
        if len(C_pruned) >= 2 and calculate_subgraph_density(C_pruned) >= density_min:
             final_clusters.append(list(C_pruned))

    return final_clusters


# --- Example Usage ---
# 1. Create a sample graph (e.g., Karate Club)
G = nx.karate_club_graph()

# 2. Run the EAGLE function (parameters need to be tuned for the specific dataset)
# Example values: density_min=0.3, pruning_min=0.5, seed_threshold=0.8
result_clusters = eagle_clustering(G, density_min=0.3, pruning_min=0.5, seed_threshold=0.8)

print(f"Total number of clusters found: {len(result_clusters)}")
for i, cluster in enumerate(result_clusters):
    # Recalculate and print density for the final cluster
    density = sum(G.subgraph(cluster).number_of_edges() for _ in [0]) / (len(cluster) * (len(cluster) - 1) / 2) if len(cluster) >= 2 else 0.0
    print(f"Cluster {i+1} (Nodes: {len(cluster)}, Density: {density:.3f}): {cluster}")