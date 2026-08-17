import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processor import DataProcessor
from gcn_lstm_model import build_model
import tensorflow as tf

def train():
    print("Initializing Data Processor...")
    processor = DataProcessor(data_dir='data')
    
    print("Loading network data...")
    G, edges_df, vertices_df = processor.load_network()
    
    print("Constructing adjacency matrix for road segments...")
    A, edge_to_idx = processor.construct_adjacency_matrix(edges_df)
    
    print("Calculating normalized laplacian...")
    laplacian = processor.calculate_normalized_laplacian(A)
    
    num_nodes = A.shape[0]
    print(f"Network has {num_nodes} road segments (nodes for GCN).")
    
    # Convert scipy laplacian to tf.sparse.SparseTensor
    coo = laplacian.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    laplacian_tf = tf.sparse.SparseTensor(indices, coo.data.astype(np.float32), coo.shape)
    # Order sparse tensor properly
    laplacian_tf = tf.sparse.reorder(laplacian_tf)
    
    print("Building GCN-LSTM model...")
    # Parameters according to paper:
    seq_len_hourly = 3 # example
    seq_len_daily = 3  # example
    num_features = 2   # e.g., vacant taxis, pickup events
    
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
    
    print("Generating synthetic training data for demonstration...")
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
    
    print("Starting training...")
    # We will train for 5 epochs for demonstration
    model.fit(train_dataset, epochs=5)
    
    print("Training complete. Saving weights...")
    os.makedirs('results', exist_ok=True)
    model.save_weights('results/gcn_lstm_weights')
    print("Model weights saved to results/gcn_lstm_weights")

if __name__ == '__main__':
    train()
