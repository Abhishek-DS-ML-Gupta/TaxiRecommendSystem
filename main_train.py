import os
import sys
import argparse
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processor import DataProcessor
from gcn_lstm_model import build_model
import tensorflow as tf

def train(max_nodes: int = 100, epochs: int = 5):
    print("Initializing Data Processor...")
    processor = DataProcessor(data_dir='data')
    
    print("Loading network data...")
    G, edges_df, vertices_df = processor.load_network()
    
    if max_nodes > 0 and max_nodes < len(edges_df):
        print(f"Selecting representative subnetwork of {max_nodes} segments for fast demonstration...")
        # Select connected subset
        edges_df = edges_df.head(max_nodes).copy()
    
    print("Constructing adjacency matrix for road segments...")
    A, edge_to_idx = processor.construct_adjacency_matrix(edges_df)
    
    print("Calculating normalized laplacian...")
    laplacian = processor.calculate_normalized_laplacian(A)
    
    num_nodes = A.shape[0]
    print(f"Network has {num_nodes} road segments (nodes for GCN).")
    
    # Convert scipy laplacian to tf.sparse.SparseTensor
    coo = laplacian.tocoo()
    indices = np.column_stack([coo.row, coo.col])
    laplacian_tf = tf.sparse.SparseTensor(indices, coo.data.astype(np.float32), coo.shape)
    laplacian_tf = tf.sparse.reorder(laplacian_tf)
    
    print("Building GCN-LSTM model...")
    seq_len_hourly = 3
    seq_len_daily = 3
    num_features = 2
    
    model = build_model(
        num_nodes=num_nodes, 
        seq_len_hourly=seq_len_hourly, 
        seq_len_daily=seq_len_daily, 
        num_features=num_features,
        laplacian_tf=laplacian_tf,
        gcn_units=32,
        lstm_units=16
    )
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mae'])
    
    print(f"Generating synthetic training data for {num_nodes} nodes...")
    batch_size = 8
    num_samples = 32
    
    h_data = np.random.rand(num_samples, seq_len_hourly, num_nodes, num_features).astype(np.float32)
    d_data = np.random.rand(num_samples, seq_len_daily, num_nodes, num_features).astype(np.float32)
    
    # Target pickup probabilities for each node
    targets = np.random.rand(num_samples, num_nodes).astype(np.float32)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {'hourly_features': h_data, 'daily_features': d_data},
        targets
    )).batch(batch_size)
    
    print(f"Starting training ({epochs} epochs)...")
    model.fit(train_dataset, epochs=epochs)
    
    print("Training complete. Saving weights...")
    os.makedirs('results', exist_ok=True)
    weights_path = os.path.join('results', 'gcn_lstm_weights.weights.h5')
    model.save_weights(weights_path)
    print(f"Model weights saved to {weights_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train GCN-LSTM Taxi Pickup Predictor")
    parser.add_argument('--max-nodes', type=int, default=100, help="Max road segments for training (default: 100 for fast demo, 0 for all 38k nodes)")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument('--full', action='store_true', help="Train on full 38,723 road network")
    args = parser.parse_args()
    
    nodes = 0 if args.full else args.max_nodes
    train(max_nodes=nodes, epochs=args.epochs)
