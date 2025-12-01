import numpy as np
from src.utils.nominals import PHANTOM_CB_NOMINALS

def get_cb_graph_edges(radius_threshold=25.0):
    """
    Defines the edges of the 'Smart Grid' for the Phantom CB points.
    Topology: 
    1. Manual Spines (Vertical + Hubs)
    2. Radius Graph (Connect any nodes closer than threshold)
    """
    edges = set() # Use set to avoid duplicates
    
    # --- 1. Manual Spines (Strong Connections) ---
    manual_edges = [
        # Left Spine
        ('CB01', 'CB02'), ('CB02', 'CB03'),
        ('CB04', 'CB05'),
        ('CB06', 'CB07'), ('CB07', 'CB08'),
        # Right Spine
        ('CB09', 'CB10'), ('CB10', 'CB11'),
        ('CB12', 'CB13'),
        ('CB14', 'CB15'), ('CB15', 'CB16'),
        # Hubs
        ('CB18', 'CB03'), ('CB18', 'CB04'),
        ('CB18', 'CB11'), ('CB18', 'CB12'),
        ('CB17', 'CB05'), ('CB17', 'CB06'),
        ('CB17', 'CB13'), ('CB17', 'CB14')
    ]
    for u, v in manual_edges:
        edges.add(tuple(sorted((u, v))))

    # --- 2. Radius Graph (Local Influence) ---
    positions = get_cb_node_positions()
    node_names = list(positions.keys())
    
    for i in range(len(node_names)):
        for j in range(i + 1, len(node_names)):
            u = node_names[i]
            v = node_names[j]
            
            p1 = np.array([positions[u]['x'], positions[u]['y'], positions[u]['z']])
            p2 = np.array([positions[v]['x'], positions[v]['y'], positions[v]['z']])
            
            dist = np.linalg.norm(p1 - p2)
            
            if dist < radius_threshold:
                edges.add(tuple(sorted((u, v))))
    
    return list(edges)

def get_cb_node_positions():
    nodes = {}
    for k, v in PHANTOM_CB_NOMINALS.items():
        if k.startswith('CB'):
            nodes[k] = {'x': v['x'], 'y': v['y'], 'z': v['z']}
    return nodes
