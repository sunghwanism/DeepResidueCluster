import networkx as nx
from itertools import combinations

def dpclus_clustering(G, d_in=0.9, cp_in=0.5):
    """
    DPClus (Density Periphery-based Clustering) 알고리즘의 핵심 로직을 구현합니다.

    Args:
        G (nx.Graph): 입력 네트워크 (networkx Graph 객체).
        d_in (float): 최소 클러스터 밀도 임계값 (d_in). 기본값 0.9.
        cp_in (float): 최소 클러스터 속성 임계값 (cp_in). 기본값 0.5.

    Returns:
        list: 각 클러스터(노드 집합)를 포함하는 리스트.
    """

    # 1. 노드 및 엣지 가중치 계산 (Common Neighbors 기반)
    # 엣지 가중치 w(u, v) = |Gamma(u) ∩ Gamma(v)|
    # 노드 가중치 W(u) = sum(w(u, v) for v in Gamma(u))

    # 비가중치 그래프를 가중치 그래프로 변환
    W_u = {}  # 노드 가중치 W(u) 저장
    G_weighted = nx.Graph(G)

    for u, v in G.edges():
        common_neighbors = len(list(nx.common_neighbors(G, u, v)))
        G_weighted.edges[u, v]['weight'] = common_neighbors

    for node in G.nodes():
        weighted_degree = sum(G_weighted.edges[node, neighbor].get('weight', 0)
                              for neighbor in G.neighbors(node))
        W_u[node] = weighted_degree
    
    # 노드 가중치 정보를 노드 속성으로 저장 (시드 선택 용이)
    nx.set_node_attributes(G_weighted, W_u, 'weight')

    # 클러스터 밀도 d(C) 계산 함수
    def calculate_density(C):
        if len(C) < 2:
            return 0.0
        
        # 엣지 가중치 합계
        sum_w = sum(G_weighted.edges[u, v]['weight']
                    for u, v in combinations(C, 2)
                    if G_weighted.has_edge(u, v))
        
        # 클러스터 C에서 가능한 최대 엣지 수 (비가중치 기준)
        max_edges = len(C) * (len(C) - 1) / 2
        
        # 밀도는 가중치 합계를 최대 엣지 수로 나눔 (DPClus 정의)
        # 단, 엣지 가중치 기반 클러스터 밀도 공식이 사용됨
        return sum_w / max_edges if max_edges > 0 else 0.0

    # 클러스터 속성 cp(v, C) 계산 함수
    def calculate_cluster_property(v, C):
        if W_u.get(v, 0) == 0:
            return 0.0

        # v와 C 내부 노드 간의 가중치 합계
        sum_w_v_C = sum(G_weighted.edges[v, u].get('weight', 0)
                        for u in C
                        if G_weighted.has_edge(v, u))

        # 노드 v의 총 가중치 (weighted degree)
        total_w_v = W_u[v]

        return sum_w_v_C / total_w_v

    # 2. 클러스터 추출 및 확장 (반복)
    
    available_nodes = set(G.nodes())
    clusters = []

    while available_nodes:
        # 2-1. 시드 선택: 남은 노드 중 가중치(W(u))가 가장 높은 노드 선택
        seed_node = max(available_nodes, key=lambda n: W_u[n], default=None)
        
        if seed_node is None:
            break

        current_cluster = {seed_node}
        available_nodes.remove(seed_node)
        
        # 2-2. 클러스터 확장
        # 확장 후보 노드: 현재 클러스터의 주변 노드 (아직 클러스터에 포함되지 않은 이웃 노드)
        while True:
            # 주변 노드 중 가장 높은 cp(v, C)를 가진 후보 선택
            periphery_candidates = {}
            
            for node in current_cluster:
                for neighbor in G.neighbors(node):
                    if neighbor in available_nodes and neighbor not in current_cluster:
                        cp_value = calculate_cluster_property(neighbor, current_cluster)
                        periphery_candidates[neighbor] = cp_value
            
            if not periphery_candidates:
                break
            
            # cp 값이 가장 높은 후보 노드 v 선택
            v_star = max(periphery_candidates, key=periphery_candidates.get)
            cp_v_star = periphery_candidates[v_star]
            
            # 클러스터에 v_star를 임시로 추가하여 밀도 계산
            C_prime = current_cluster.union({v_star})
            d_C_prime = calculate_density(C_prime)

            # 2-3. 조건 확인 및 확장
            # (1) 밀도 조건: d(C') >= d_in
            # (2) 클러스터 속성 조건: cp(v, C) >= cp_in
            if d_C_prime >= d_in and cp_v_star >= cp_in:
                # 조건 만족 시, 클러스터 확장 및 노드 제거
                current_cluster.add(v_star)
                available_nodes.remove(v_star)
            else:
                # 조건을 만족하는 노드가 없거나, 가장 좋은 노드가 조건을 만족하지 못하면 확장 중단
                break

        # 3. 클러스터 저장
        clusters.append(current_cluster)

    return [list(c) for c in clusters]




    
# import networkx as nx
# from itertools import combinations

# def dpclus_clustering(G, d_in=0.9, cp_in=0.5):
#     """
#     Implements the core logic of the DPClus (Density Periphery-based Clustering) algorithm.

#     DPClus uses a seed-extension approach, prioritizing nodes that maintain a high 
#     cluster density and strong connectivity to the current cluster.

#     Args:
#         G (nx.Graph): The input network (networkx Graph object). Assumes an unweighted graph,
#                       which will be converted to a weighted graph based on common neighbors.
#         d_in (float): Minimum cluster density threshold (d_in). Default is 0.9.
#         cp_in (float): Minimum cluster property threshold (cp_in). Default is 0.5.

#     Returns:
#         list: A list where each element is a list of nodes belonging to a cluster.
#     """

#     # 1. Calculate Node and Edge Weights (Based on Common Neighbors)
#     # Edge Weight w(u, v) = |Gamma(u) ∩ Gamma(v)|
#     # Node Weight W(u) = sum(w(u, v) for v in Gamma(u)) (Weighted Degree)

#     G_weighted = nx.Graph(G)
#     W_u = {}  # Stores node weights W(u)

#     # Calculate edge weights w(u, v)
#     for u, v in G.edges():
#         common_neighbors = len(list(nx.common_neighbors(G, u, v)))
#         G_weighted.edges[u, v]['weight'] = common_neighbors

#     # Calculate node weights W(u) (Weighted Degree)
#     for node in G.nodes():
#         weighted_degree = sum(G_weighted.edges[node, neighbor].get('weight', 0)
#                               for neighbor in G.neighbors(node))
#         W_u[node] = weighted_degree
    
#     # Store node weights as node attributes (for easier seed selection)
#     nx.set_node_attributes(G_weighted, W_u, 'weight')

#     # Function to calculate Cluster Density d(C)
#     # d(C) = (Sum of weighted edges in C) / (Max possible edges in C)
#     def calculate_density(C):
#         if len(C) < 2:
#             return 0.0
        
#         # Sum of weighted edges within cluster C
#         sum_w = sum(G_weighted.edges[u, v]['weight']
#                     for u, v in combinations(C, 2)
#                     if G_weighted.has_edge(u, v))
        
#         # Maximum number of possible edges in C
#         max_edges = len(C) * (len(C) - 1) / 2
        
#         return sum_w / max_edges if max_edges > 0 else 0.0

#     # Function to calculate Cluster Property cp(v, C)
#     # cp(v, C) = (Sum of weighted edges between v and C) / (Total weighted degree of v)
#     def calculate_cluster_property(v, C):
#         if W_u.get(v, 0) == 0:
#             return 0.0

#         # Sum of weights between node v and nodes in cluster C
#         sum_w_v_C = sum(G_weighted.edges[v, u].get('weight', 0)
#                         for u in C
#                         if G_weighted.has_edge(v, u))

#         # Total weighted degree of node v
#         total_w_v = W_u[v]

#         return sum_w_v_C / total_w_v

#     # 2. Iterative Cluster Extraction and Expansion
    
#     available_nodes = set(G.nodes())
#     clusters = []

#     while available_nodes:
#         # 2-1. Seed Selection: Choose the node with the highest weight W(u) among available nodes
#         seed_node = max(available_nodes, key=lambda n: W_u[n], default=None)
        
#         if seed_node is None:
#             break

#         current_cluster = {seed_node}
#         available_nodes.remove(seed_node)
        
#         # 2-2. Cluster Expansion Loop
#         while True:
#             # Identify candidate nodes in the periphery (neighbors of current cluster, not yet clustered)
#             periphery_candidates = {}
            
#             for node in current_cluster:
#                 for neighbor in G.neighbors(node):
#                     if neighbor in available_nodes and neighbor not in current_cluster:
#                         # Calculate cluster property for the candidate
#                         cp_value = calculate_cluster_property(neighbor, current_cluster)
#                         periphery_candidates[neighbor] = cp_value
            
#             if not periphery_candidates:
#                 break
            
#             # Select the candidate node v_star with the highest cp value
#             v_star = max(periphery_candidates, key=periphery_candidates.get)
#             cp_v_star = periphery_candidates[v_star]
            
#             # Temporarily add v_star to calculate the resulting density
#             C_prime = current_cluster.union({v_star})
#             d_C_prime = calculate_density(C_prime)

#             # 2-3. Check Conditions and Expand
#             # (1) Density Condition: d(C') >= d_in (Must maintain minimum density)
#             # (2) Cluster Property Condition: cp(v, C) >= cp_in (Must have strong connection to C)
#             if d_C_prime >= d_in and cp_v_star >= cp_in:
#                 # If conditions are met, expand the cluster and remove the node from available set
#                 current_cluster.add(v_star)
#                 available_nodes.remove(v_star)
#             else:
#                 # If the best candidate fails the check, stop expansion for this cluster
#                 break

#         # 3. Store the final cluster
#         clusters.append(current_cluster)

#     return [list(c) for c in clusters]


# # --- Example Usage ---
# # 1. Create a sample graph (e.g., Karate Club)
# G = nx.karate_club_graph()

# # 2. Run the DPClus function (parameters d_in and cp_in should be tuned experimentally)
# result_clusters = dpclus_clustering(G, d_in=0.7, cp_in=0.5)

# print(f"Total number of clusters found: {len(result_clusters)}")
# for i, cluster in enumerate(result_clusters):
#     print(f"Cluster {i+1} (Nodes: {len(cluster)}): {cluster}")

