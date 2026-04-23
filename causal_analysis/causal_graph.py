"""
File Name: causal_graph.py

Purpose: This module builds and visualizes directed graphs based on
causation entropy results. 
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from pipeline.visualization import save_figure
#=================================================================================
# configuration for reproducibility
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
#=================================================================================

#=================================================================================
def build_causal_graph(results):
    """
    Builds a causal graph (edge list) from causation entropy (CSE) 
    results. The nodes in this graph are the selected features and the 
    target variable. The edges are defined as the path from feature to 
    target and from an earlier feature to a later feature (causal layering).

    Returns:
        - edges (list of dict): source, target, weight
    """
    edges = []

    selected = results["selected_features"]
    scores = results["scores"]
    p_vals = results["p_values"]
    stability = results["stability"]

    target = "damage_binary"

    for i, feature in enumerate(selected):
        edges.append({
            "source": feature,
            "target": target, 
            "weight": scores[feature],
            "p_value": p_vals.get(feature, None),
            "type": "conditional_dependency"
        })

        # edges between features (causal layering)
        for prev in selected[:i]:
            edges.append({
                "source": prev,
                "target": feature,
                "weight": scores[feature],
                "type": "conditional_dependency"
        })

    return edges
#=================================================================================

#=================================================================================
def plot_graph(edges, save = False):
    """
    Visualizes directed graphs for causation entropy pipeline.
    """
    G = nx.DiGraph()

    for e in edges:
        G.add_edge(
            e["source"],
            e["target"],
            weight = e.get("weight", 0)
        )

    plt.figure(figsize = (10,6))
    pos = nx.spring_layout(G, seed = RANDOM_SEED)

    nx.draw(G, pos, with_labels = True, node_size = 2000, font_size = 10)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)

    plt.title("Causal Graph (Causation Entropy)")
    
    if save:
        save_figure("causal_graph")

    plt.show()
#=================================================================================
