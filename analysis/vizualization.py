
import ast
import networkx as nx
import matplotlib.patches as mpatches

import numpy as np
from collections import Counter

import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.multitest import multipletests


def plot_uniprot_counts(counts_dict, top_n=20, saveFileName=None):
    sorted_items = sorted(counts_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    ids = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    plt.figure(figsize=(15, 5))
    bars = plt.bar(ids, counts, color='skyblue', edgecolor='navy')

    plt.title(f'Top {top_n} Most Frequent Gene Name', fontsize=15)
    plt.xlabel('UniProt ID', fontsize=12)
    plt.ylabel('Frequency (Count)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, yval, ha='center', va='bottom')

    plt.tight_layout()
    if saveFileName:
        plt.savefig(f'{saveFileName}.png')
    plt.show()


def pathogenicity_graph_viz(G, df, gene_dict, min_prot=3, limit_n_clusters=None, saveFileName=None):
    H = G.copy()

    for c, (idx, row) in enumerate(df.iterrows()):
        if limit_n_clusters:
            if c >= limit_n_clusters:
                break
        
        node_list_in_clusters = row['nodes']
        cluster_id = row['cluster_id']
        nodes = ast.literal_eval(node_list_in_clusters)

        group_map = {}
        for node in nodes:
            uni_id = node.split("_")[0]
            if uni_id not in group_map:
                group_map[uni_id] = []
            group_map[uni_id].append(node)
        
        if len(group_map.keys()) >= min_prot:
            temp_G = G.subgraph(nodes).copy()
            
            for uni_id, member_nodes in group_map.items():
                center_node = member_nodes[0]
                for other_node in member_nodes[1:]:
                    temp_G = nx.contracted_nodes(temp_G, center_node, other_node, self_loops=False)
                
                mapping = {center_node: uni_id}
                temp_G = nx.relabel_nodes(temp_G, mapping)

            uniprot_list = [n.split("_")[0] for n in nodes]
            uniprot_counts = Counter(uniprot_list)
            
            plt.figure(figsize=(16, 10)) 
            pos = nx.spring_layout(temp_G, seed=42)

            unique_ids = list(temp_G.nodes())
            cmap = cm.get_cmap('tab20', len(unique_ids))
            
            node_color_list = []
            legend_handles = []
            
            custom_labels = {}
            
            for i, node in enumerate(unique_ids):
                color = cmap(i)
                node_color_list.append(color)
                
                gene_name = gene_dict.get(node.split('-')[0], node)
                count = uniprot_counts[node]
                
                custom_labels[node] = f"{node}\n({gene_name})"
                
                legend_label = f"{gene_name} (n={count})"
                patch = mpatches.Patch(color=color, label=legend_label)
                legend_handles.append(patch)

            node_sizes = [uniprot_counts[node] * 30 for node in unique_ids] 
            
            nx.draw(temp_G, pos, 
                    with_labels=False, 
                    node_size=node_sizes,
                    node_color=node_color_list,
                    edge_color='gray',
                    alpha=0.9)

            nx.draw_networkx_labels(temp_G, pos, 
                                labels=custom_labels,
                                font_size=9,
                                font_color='black',
                                font_weight='bold')
            
            plt.legend(handles=legend_handles, 
                    title="Gene Name (Count)", 
                    bbox_to_anchor=(1.05, 1), 
                    loc='upper left',
                    fontsize=10)
            
            plt.title(f"Cluster {idx} (Merged by Protein ID)", fontsize=20)
            plt.tight_layout()
            if saveFileName:
                plt.savefig(f'{saveFileName}_{cluster_id}.png')
            plt.show()


def plot_topological_analysis(df, features, target):
    stats_results = []
    
    for feature in features:
        temp_df = df[[feature, target]].dropna()
        x, y = temp_df[feature], temp_df[target]
        slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
        stats_results.append({'feature': feature, 'r': r_value, 'p': p_value, 
                              'slope': slope, 'intercept': intercept, 'x': x, 'y': y})

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 6)
    axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6]),
            fig.add_subplot(gs[1, 1:3]), fig.add_subplot(gs[1, 3:5])]
    
    for i, res in enumerate(stats_results):
        ax = axes[i]
        x, y = res['x'], res['y']
        
        sns.scatterplot(x=x, y=y, ax=ax, alpha=0.4, edgecolor=None, color='royalblue')
        line = res['slope'] * x + res['intercept']
        ax.plot(x, line, color='red', linewidth=1.5)

        stats_text = (f"$r$ = {res['r']:.3f}\n"
                      f"$p$ = {res['p']:.3e}")
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgrey')
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        ax.set_title(f"{target} vs {res['feature'].upper()}", fontsize=8, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()  
    plt.show()