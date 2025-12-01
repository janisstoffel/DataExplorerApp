import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import glob
import sys
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# Add the project root to the python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.gnn_model import SimpleGNN, create_graph_data
from src.utils.graph_generator import get_cb_node_positions

def load_training_data(data_dir):
    """
    Reads all VIANT_PHANTOM_*.csv files in the data directory recursively.
    Returns a list of graph snapshots (Data objects).
    """
    print(f"Searching for training data in {data_dir}...")
    
    # Get base graph structure (Nominal Data)
    base_data, node_names = create_graph_data()
    nominal_pos = base_data.x # [Num_Nodes, 3]
    edge_index = base_data.edge_index
    
    dataset = []
    
    # Recursive search for CSV files
    pattern = os.path.join(data_dir, "**", "VIANT_PHANTOM_*.csv")
    csv_files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(csv_files)} potential training files.")
    
    for file_path in csv_files:
        try:
            # The CSVs have 3 lines of metadata before the header
            # TrackerID,Phantom
            # BatchNumber,...
            # BenchNumber,...
            # Feature,X,Y,Z  <-- This is the header (line 4, index 3)
            df = pd.read_csv(file_path, header=3)
            
            # Check if it has the expected structure
            # Rows: Feature (CB01...), Columns: X, Y, Z
            # Note: Column names might have whitespace
            df.columns = [c.strip() for c in df.columns]
            
            if 'Feature' not in df.columns or 'X' not in df.columns:
                # Try reading without skipping if the format is different
                df = pd.read_csv(file_path)
                df.columns = [c.strip() for c in df.columns]
                if 'Feature' not in df.columns or 'X' not in df.columns:
                    continue
                
            # Create a dictionary for this snapshot: {CB01: [x,y,z], ...}
            snapshot_pos = {}
            for _, row in df.iterrows():
                feat = row['Feature']
                if isinstance(feat, str) and feat.startswith('CB'):
                    snapshot_pos[feat] = [row['X'], row['Y'], row['Z']]
            
            # Construct the target tensor in the same order as node_names
            target_list = []
            valid_snapshot = True
            for name in node_names:
                if name in snapshot_pos:
                    target_list.append(snapshot_pos[name])
                else:
                    # Missing data for a node in this snapshot
                    valid_snapshot = False
                    break
            
            if valid_snapshot:
                target_tensor = torch.tensor(target_list, dtype=torch.float)
                
                # Input: Nominal Positions
                # Target: Actual Positions from CSV
                data = Data(x=nominal_pos, edge_index=edge_index, y=target_tensor)
                dataset.append(data)
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    print(f"Successfully loaded {len(dataset)} valid training snapshots.")
    return dataset

def train_model():
    # Paths
    # __file__ = .../Data_Explorer_App/src/utils/train_gnn.py
    # Go up 4 levels to get to CSVDATA root
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    DATA_DIR = os.path.join(BASE_DIR, 'CSV Data')
    MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'phantom_gnn_model.pth')
    
    # Hyperparameters
    EPOCHS = 200
    LEARNING_RATE = 0.001
    HIDDEN_DIM = 64
    
    # 1. Load Data
    dataset = load_training_data(DATA_DIR)
    
    if len(dataset) == 0:
        print("No training data found. Aborting.")
        return

    # Split
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    # 2. Initialize Model
    # Input: 3 (Nominal X,Y,Z)
    # Output: 3 (Actual X,Y,Z)
    model = SimpleGNN(num_node_features=3, hidden_channels=HIDDEN_DIM, num_classes=3)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    print("Starting training...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for data in train_data:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_data)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}")
            
    # 3. Save
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
