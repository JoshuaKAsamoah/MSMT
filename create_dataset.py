import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import pickle

def load_pems04_data(data_dir):
    """Load the original PEMS04 dataset"""
    train_data = np.load(os.path.join(data_dir, 'train.npz'))
    val_data = np.load(os.path.join(data_dir, 'val.npz'))
    test_data = np.load(os.path.join(data_dir, 'test.npz'))
    
    # Extract the data arrays
    train_x = train_data['x']  # Shape: (samples, input_len, nodes, features)
    train_y = train_data['y']  # Shape: (samples, output_len, nodes, features)
    val_x = val_data['x']
    val_y = val_data['y']
    test_x = test_data['x']
    test_y = test_data['y']
    
    print(f"Original PEMS04 shapes:")
    print(f"Train X: {train_x.shape}, Train Y: {train_y.shape}")
    print(f"Val X: {val_x.shape}, Val Y: {val_y.shape}")
    print(f"Test X: {test_x.shape}, Test Y: {test_y.shape}")
    
    return train_x, train_y, val_x, val_y, test_x, test_y

def reconstruct_continuous_data(train_x, train_y, val_x, val_y, test_x, test_y):
    """
    Reconstruct continuous time series from the split datasets
    This assumes the original data was split sequentially
    """
    # Get dimensions
    input_len = train_x.shape[1]  # Original input length (e.g., 12)
    output_len = train_y.shape[1]  # Original output length (e.g., 12)
    num_nodes = train_x.shape[2]
    num_features = train_x.shape[3]
    
    # Calculate total length needed
    total_samples = len(train_x) + len(val_x) + len(test_x)
    total_timesteps = total_samples + input_len + output_len - 1
    
    # Initialize continuous data array
    continuous_data = np.zeros((total_timesteps, num_nodes, num_features))
    
    # Fill in the data
    # Start with the first train sample's input
    continuous_data[:input_len] = train_x[0]
    
    current_pos = input_len
    
    # Add train targets
    for i in range(len(train_x)):
        if i == 0:
            continuous_data[current_pos:current_pos + output_len] = train_y[i]
            current_pos += output_len
        else:
            # Only add the last time step to avoid overlap
            continuous_data[current_pos] = train_y[i][-1]
            current_pos += 1
    
    # Add val targets
    for i in range(len(val_x)):
        continuous_data[current_pos] = val_y[i][-1]
        current_pos += 1
    
    # Add test targets  
    for i in range(len(test_x)):
        continuous_data[current_pos] = test_y[i][-1]
        current_pos += 1
    
    return continuous_data

def create_sequences_60(data, input_len=60, output_len=60):
    """
    Create 60-step input and output sequences
    """
    num_samples = data.shape[0] - input_len - output_len + 1
    if num_samples <= 0:
        raise ValueError(f"Not enough data points. Need at least {input_len + output_len}, got {data.shape[0]}")
    
    X = np.zeros((num_samples, input_len, data.shape[1], data.shape[2]))
    Y = np.zeros((num_samples, output_len, data.shape[1], data.shape[2]))
    
    for i in range(num_samples):
        X[i] = data[i:i + input_len]
        Y[i] = data[i + input_len:i + input_len + output_len]
    
    return X, Y

def split_data(X, Y, train_ratio=0.6, val_ratio=0.2):
    """Split data into train/val/test sets"""
    num_samples = len(X)
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    
    # Sequential split (important for time series)
    train_x = X[:train_size]
    train_y = Y[:train_size]
    
    val_x = X[train_size:train_size + val_size]
    val_y = Y[train_size:train_size + val_size]
    
    test_x = X[train_size + val_size:]
    test_y = Y[train_size + val_size:]
    
    return train_x, train_y, val_x, val_y, test_x, test_y

def create_pems04_60_dataset(input_dir, output_dir):
    """
    Create PEMS04_60 dataset from original PEMS04 data
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original data
    train_x_orig, train_y_orig, val_x_orig, val_y_orig, test_x_orig, test_y_orig = load_pems04_data(input_dir)
    
    # Reconstruct continuous time series
    print("Reconstructing continuous time series...")
    continuous_data = reconstruct_continuous_data(
        train_x_orig, train_y_orig, val_x_orig, val_y_orig, test_x_orig, test_y_orig
    )
    print(f"Continuous data shape: {continuous_data.shape}")
    
    # Create 60-step sequences
    print("Creating 60-step sequences...")
    X_60, Y_60 = create_sequences_60(continuous_data, input_len=60, output_len=60)
    print(f"New sequences - X: {X_60.shape}, Y: {Y_60.shape}")
    
    # Split into train/val/test
    print("Splitting data...")
    train_x_60, train_y_60, val_x_60, val_y_60, test_x_60, test_y_60 = split_data(X_60, Y_60)
    
    print(f"Final shapes:")
    print(f"Train: X={train_x_60.shape}, Y={train_y_60.shape}")
    print(f"Val: X={val_x_60.shape}, Y={val_y_60.shape}")
    print(f"Test: X={test_x_60.shape}, Y={test_y_60.shape}")
    
    # Save the datasets
    print("Saving datasets...")
    np.savez_compressed(
        os.path.join(output_dir, 'train.npz'),
        x=train_x_60,
        y=train_y_60
    )
    
    np.savez_compressed(
        os.path.join(output_dir, 'val.npz'),
        x=val_x_60,
        y=val_y_60
    )
    
    np.savez_compressed(
        os.path.join(output_dir, 'test.npz'),
        x=test_x_60,
        y=test_y_60
    )
    
    # Create adjacency matrix (copy from original)
    if os.path.exists(os.path.join(input_dir, 'adj_PEMS04.pkl')):
        import shutil
        shutil.copy(
            os.path.join(input_dir, 'adj_PEMS04.pkl'),
            os.path.join(output_dir, 'adj_PEMS04.pkl')
        )
        print("Copied adjacency matrix")
    
    print(f"PEMS04_60 dataset created successfully in {output_dir}")

# Alternative approach: Load raw data if available
def create_from_raw_data(raw_data_path, output_dir):
    """
    If you have the raw continuous PEMS04 data (not preprocessed)
    """
    # Load raw data (adjust this based on your raw data format)
    # This could be a CSV, NPZ, or other format
    raw_data = np.load(raw_data_path)  # Shape: (timesteps, nodes, features)
    
    print(f"Raw data shape: {raw_data.shape}")
    
    # Create 60-step sequences
    X_60, Y_60 = create_sequences_60(raw_data, input_len=60, output_len=60)
    
    # Split data
    train_x_60, train_y_60, val_x_60, val_y_60, test_x_60, test_y_60 = split_data(X_60, Y_60)
    
    # Save datasets (same as above)
    os.makedirs(output_dir, exist_ok=True)
    
    np.savez_compressed(os.path.join(output_dir, 'train.npz'), x=train_x_60, y=train_y_60)
    np.savez_compressed(os.path.join(output_dir, 'val.npz'), x=val_x_60, y=val_y_60)
    np.savez_compressed(os.path.join(output_dir, 'test.npz'), x=test_x_60, y=test_y_60)
    
    print(f"PEMS04_60 dataset created from raw data in {output_dir}")

# Usage example
if __name__ == "__main__":
    # Method 1: Create from existing preprocessed PEMS04 data
    input_directory = "data/PEMS04"  # Your current PEMS04 directory
    output_directory = "data/PEMS04_60"  # New directory for 60-step data
    
    create_pems04_60_dataset(input_directory, output_directory)
    
    # Method 2: If you have raw continuous data (uncomment if needed)
    # create_from_raw_data("pems08.npz", output_directory)