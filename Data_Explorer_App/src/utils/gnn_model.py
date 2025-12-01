import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
from .graph_generator import get_cb_graph_edges, get_cb_node_positions

class SimpleGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def create_graph_data():
    """
    Creates a PyTorch Geometric Data object representing the Phantom CB grid.
    """
    positions = get_cb_node_positions()
    edges = get_cb_graph_edges()
    
    # Create a mapping from node name to index
    node_names = sorted(list(positions.keys()))
    node_to_idx = {name: i for i, name in enumerate(node_names)}
    
    # Node features: [x, y, z] coordinates
    # We normalize them slightly for better ML performance usually, but raw is fine for demo
    x_features = []
    for name in node_names:
        pos = positions[name]
        x_features.append([pos['x'], pos['y'], pos['z']])
    
    x = torch.tensor(x_features, dtype=torch.float)
    
    # Edge index
    edge_indices = []
    for start, end in edges:
        if start in node_to_idx and end in node_to_idx:
            src = node_to_idx[start]
            dst = node_to_idx[end]
            # Add bidirectional edges for undirected graph
            edge_indices.append([src, dst])
            edge_indices.append([dst, src])
            
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    data = Data(x=x, edge_index=edge_index)
    return data, node_names

def predict_fields_mock(input_data=None):
    """
    Runs a forward pass of the GNN.
    If a trained model file exists, it loads it.
    Otherwise, it initializes a random model.
    """
    import os
    
    data, node_names = create_graph_data()
    
    # Initialize model
    # Input features: 3 (x, y, z)
    # Hidden: 64
    # Output: 3 (Predicted X, Y, Z)
    model = SimpleGNN(num_node_features=3, hidden_channels=64, num_classes=3)
    
    # Load weights if they exist
    model_path = os.path.join(os.path.dirname(__file__), 'phantom_gnn_model.pth')
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
            model.eval()
            print("Loaded trained Phantom GNN model.")
        except Exception as e:
            print(f"Failed to load model: {e}. Using random weights.")
    else:
        print("No trained model found. Using random weights (untrained).")
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    
    # Map results back to node names
    # We calculate the displacement magnitude (Actual - Nominal) for visualization
    predictions = {}
    for i, name in enumerate(node_names):
        # out[i] is the predicted position [x, y, z]
        # data.x[i] is the nominal position [x, y, z]
        predicted_pos = out[i]
        nominal_pos = data.x[i]
        
        # Calculate displacement (error/deformation)
        displacement = torch.norm(predicted_pos - nominal_pos).item()
        predictions[name] = displacement
        
    return predictions
