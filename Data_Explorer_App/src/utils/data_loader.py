import pandas as pd
import os
import sys

# Determine the base path
# If frozen (exe), use the executable's directory.
# If script, use the script's directory or CWD.
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    # For development, we might want to keep the hardcoded path or use CWD
    # defaulting to CWD for portability during dev
    BASE_DIR = os.getcwd()

# Fallback for specific dev environment if needed, otherwise use BASE_DIR
DATA_DIR = BASE_DIR 

def get_csv_files(folder_path=None):
    """
    Get list of CSV files in the data directory.
    If folder_path is provided, looks inside that subfolder of DATA_DIR.
    Otherwise looks in DATA_DIR root.
    """
    target_dir = DATA_DIR
    if folder_path:
        target_dir = os.path.join(DATA_DIR, folder_path)
    
    if not os.path.exists(target_dir):
        return []
        
    files = [f for f in os.listdir(target_dir) if f.endswith('.csv')]
    return sorted(files)

def load_csv_data(filename, folder_path=None):
    """
    Load a specific CSV file.
    """
    target_dir = DATA_DIR
    if folder_path:
        target_dir = os.path.join(DATA_DIR, folder_path)
        
    file_path = os.path.join(target_dir, filename)
    
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return pd.DataFrame()

def parse_timestamp(filename, prefix="VIANT_PHANTOM_"):
    # Expected format: PREFIX_YYYYMMDDHHMMSS.csv
    # Escape the prefix for regex safety
    escaped_prefix = re.escape(prefix)
    # Use raw string for regex pattern to avoid warnings
    pattern = rf'{escaped_prefix}(\d{{14}})\.csv'
    match = re.search(pattern, filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
    return None

def load_cs_data_robust(file_prefix="VIANT_PHANTOM_"):
    data = []
    
    # Use DATA_DIR instead of hardcoded CSV_ROOT
    search_root = DATA_DIR
    
    for root, dirs, files in os.walk(search_root):
        for file in files:
            if file.startswith(file_prefix) and file.endswith(".csv"):
                file_path = os.path.join(root, file)
                timestamp = parse_timestamp(file, prefix=file_prefix)
                
                if timestamp:
                    try:
                        # Read metadata first
                        with open(file_path, 'r') as f:
                            lines = f.readlines()
                        
                        batch_number = "Unknown"
                        start_line = 0
                        found_header = False
                        
                        for i, line in enumerate(lines):
                            if line.startswith("BatchNumber"):
                                parts = line.strip().split(',')
                                if len(parts) > 1:
                                    batch_number = parts[1]
                            if line.startswith("Feature,X,Y,Z"):
                                start_line = i
                                found_header = True
                                break
                        
                        # Read data table
                        if found_header:
                            df = pd.read_csv(file_path, skiprows=start_line)
                            
                            # We load all points that have valid coordinates
                            if 'Feature' in df.columns and 'X' in df.columns:
                                # Get relative folder path
                                rel_folder = os.path.relpath(root, search_root)
                                if rel_folder == ".":
                                    rel_folder = "Root"
                                
                                for _, row in df.iterrows():
                                    # Filter for CS and CB points
                                    feature = str(row['Feature'])
                                    if (feature.startswith('CS') or feature.startswith('CB')) and \
                                       pd.notna(row['X']) and pd.notna(row['Y']) and pd.notna(row['Z']):
                                        data.append({
                                            'BatchNumber': batch_number,
                                            'Timestamp': timestamp,
                                            'Feature': feature,
                                            'X': float(row['X']),
                                            'Y': float(row['Y']),
                                            'Z': float(row['Z']),
                                            'FileName': file,
                                            'Folder': rel_folder
                                        })
                                
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
                        
    return pd.DataFrame(data)
