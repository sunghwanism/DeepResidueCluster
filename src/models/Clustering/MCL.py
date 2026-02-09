import os
import networkx as nx
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import norm
from scipy.sparse.csgraph import connected_components
from typing import Dict, Set, Any, List, Tuple

NUM_THREADS = os.cpu_count()
if NUM_THREADS:
    os.environ['OMP_NUM_THREADS'] = str(NUM_THREADS)
    os.environ['MKL_NUM_THREADS'] = str(NUM_THREADS)

def parallel_mcl_clustering(
    graph: nx.Graph,
    inflation_power: float = 5.0,
    expansion_power: int = 3,
    max_iter: int = 1000,
    pruning_threshold: float = 0.13,
    convergence_threshold: float = 1e-6,
    use_weights: bool = False,
    weight_key: str = 'weight',
    singleton_clusters: bool = True,
    min_cluster_size: int = 2,
    add_self_loops: bool = True
) -> Tuple[Dict[int, Set[Any]], Dict[int, Dict[str, Any]], Dict[int, List[Tuple[Any, Any, float]]], Any]:
    
    # 1. Input Validation
    if not isinstance(graph, nx.Graph):
        raise TypeError("Input must be a NetworkX graph.")

    nodelist = list(graph.nodes())
    num_nodes = len(nodelist)
    
    index_to_node = {i: node for i, node in enumerate(nodelist)}

    weight_arg = weight_key if use_weights else None

    # 2. Matrix Initialization
    matrix = nx.to_scipy_sparse_array(
        graph, 
        nodelist=nodelist, 
        weight=weight_arg, 
        dtype=np.float64, 
        format="csr"
    )

    if use_weights and matrix.data.size > 0:
        if np.any(matrix.data < 0):
            matrix.data = np.abs(matrix.data)
            matrix.eliminate_zeros()

    if add_self_loops:
        matrix = matrix + eye(num_nodes, format='csr')

    def normalize_matrix(mat):
        col_sums = np.array(mat.sum(axis=0)).flatten()
        col_sums[col_sums == 0] = 1.0
        D_inv = diags(1.0 / col_sums, format="csr")
        return mat @ D_inv

    M = normalize_matrix(matrix)

    # 3. MCL Iteration
    for i in range(max_iter):
        M_prev = M.copy()
        
        # Expansion
        if expansion_power == 2:
            M = M @ M
        else:
            M = M.power(expansion_power)
        
        # Inflation
        M.data = np.power(M.data, inflation_power)
        M = normalize_matrix(M)
        
        # Pruning
        if pruning_threshold > 0:
            mask = M.data >= pruning_threshold
            if not np.all(mask):
                M.data = M.data * mask
                M.eliminate_zeros()

        if norm(M - M_prev) < convergence_threshold:
            break

    # ---------------------------- 4. Cluster Extraction (Scipy 최적화 + 원본 로직 호환) ------------------------
    
    n_components, labels = connected_components(
        csgraph=M, 
        directed=False, 
        connection='strong', 
        return_labels=True
    )

    clusters = {}
    cluster_info = {}
    cluster_edges = {}
    
    diag_array = M.diagonal()
    
    # 그룹핑
    temp_groups = {}
    for idx, label in enumerate(labels):
        if label not in temp_groups:
            temp_groups[label] = []
        temp_groups[label].append(idx)

    final_cid = 0
    node_idx_to_cid = {}

    for label, indices in temp_groups.items():
        if len(indices) < min_cluster_size:
            continue
        if not singleton_clusters and len(indices) == 1:
            continue
            
        cid = final_cid
        
        node_names = {index_to_node[idx] for idx in indices}
        clusters[cid] = node_names
        
        attractors = []
        node_stability = {}
        
        for idx in indices:
            node_idx_to_cid[idx] = cid
            name = index_to_node[idx]
            
            if diag_array[idx] > 0:
                attractors.append(name)
            
            col_vals = M[:, idx].data
            stablity = np.max(col_vals) if col_vals.size > 0 else 0.0
            node_stability[name] = float(stablity)
            
        cluster_info[cid] = {
            "attractors": attractors,
            "node_stability": node_stability
        }
        cluster_edges[cid] = []
        final_cid += 1

    # 5. Edge Extraction (Column-based)
    M_coo = M.tocoo()
    for r, c, w in zip(M_coo.row, M_coo.col, M_coo.data):
        if r in node_idx_to_cid and c in node_idx_to_cid:
            cid_r = node_idx_to_cid[r]
            cid_c = node_idx_to_cid[c]
            
            if cid_r == cid_c:
                src = index_to_node[c]
                dst = index_to_node[r]
                # (Source, Target, Weight)
                cluster_edges[cid_r].append((src, dst, float(w)))

    return clusters, cluster_info, cluster_edges, M