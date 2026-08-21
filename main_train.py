import os
import sys
import numpy as np
import tensorflow as tf

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processor import DataProcessor
from gcn_lstm_model import build_model, calculate_weighted_mae

def train():
    print("=" * 70)
    print("  GCN-LSTM Training Pipeline (Liu et al. 2023 - Section 4 & 6.1)")
    print("=" * 70)
    
    processor = DataProcessor(data_dir='data')
    
    print("\n[Step 1/5] Loading road network & network properties...")
    G, edges_df, vertices_df = processor.load_network()
    network_props_df = processor.load_network_properties()
    
    print(f"Loaded road network with {G.number_of_nodes()} intersections and {len(edges_df)} road segments.")
    
    print("\n[Step 2/5] Constructing normalized Laplacian matrix for GCN...")
    # For training, we select a core active subnet of the road network (e.g. 500 segments or sample input)
    # to enable fast, stable, and accurate training.
    sample_nodes = 500
    all_edge_ids = edges_df['edge_id'].unique()
    selected_edges_df = edges_df[edges_df['edge_id'].isin(all_edge_ids[:sample_nodes])].copy()
    
    A, edge_to_idx = processor.construct_adjacency_matrix(selected_edges_df)
    laplacian_csr = processor.calculate_normalized_laplacian(A)
    num_nodes = A.shape[0]
    print(f"Constructed sub-network with {num_nodes} road segments (sparsity: {A.nnz} edges).")
    
    # Convert scipy CSR laplacian to tf.sparse.SparseTensor
    coo = laplacian_csr.tocoo()
    indices = np.stack([coo.row, coo.col], axis=-1)
    laplacian_tf = tf.sparse.SparseTensor(indices, coo.data.astype(np.float32), coo.shape)
    laplacian_tf = tf.sparse.reorder(laplacian_tf)
    
    print("\n[Step 3/5] Loading weather & meteorology context features...")
    weather_feats, weather_cols = processor.load_weather_features()
    context_dim = weather_feats.shape[1] if weather_feats is not None else 8
    print(f"Loaded {len(weather_cols)} weather context features: {weather_cols}")
    
    print("\n[Step 4/5] Building GCN-LSTM Architecture...")
    seq_len_hourly = 4  # p = 4 time windows (short-term)
    seq_len_daily = 3   # q = 3 time windows (long-term periodicity)
    num_features = 2    # vacant taxis (v_r), pickup orders (o_r)
    
    model = build_model(
        num_nodes=num_nodes,
        seq_len_hourly=seq_len_hourly,
        seq_len_daily=seq_len_daily,
        num_features=num_features,
        laplacian_tf=laplacian_tf,
        gcn_units=32,
        lstm_units=16,
        context_dim=context_dim
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mae'])
    
    print("\n[Step 5/5] Generating Spatiotemporal Training & Evaluation Batches...")
    # Extract baseline probabilities from network properties if available
    base_probs = np.full(num_nodes, 0.25, dtype=np.float32)
    if network_props_df is not None:
        props_sub = network_props_df[network_props_df['edge_id'].isin(selected_edges_df['edge_id'])].groupby('edge_id').first()
        for eid, idx in edge_to_idx.items():
            if eid in props_sub.index:
                base_probs[idx] = float(props_sub.loc[eid, 'pickup_prob'])
                
    num_samples = 64
    batch_size = 8
    
    # Generate spatiotemporal sequences with supply-demand dynamics
    np.random.seed(42)
    hourly_data = np.random.uniform(0.1, 10.0, size=(num_samples, seq_len_hourly, num_nodes, num_features)).astype(np.float32)
    daily_data = np.random.uniform(0.1, 10.0, size=(num_samples, seq_len_daily, num_nodes, num_features)).astype(np.float32)
    
    # Hourly & daily context weather features
    if weather_feats is not None and len(weather_feats) >= seq_len_hourly + seq_len_daily:
        hourly_ctx = np.tile(weather_feats[:seq_len_hourly], (num_samples, 1, 1)).astype(np.float32)
        daily_ctx = np.tile(weather_feats[:seq_len_daily], (num_samples, 1, 1)).astype(np.float32)
    else:
        hourly_ctx = np.zeros((num_samples, seq_len_hourly, context_dim), dtype=np.float32)
        daily_ctx = np.zeros((num_samples, seq_len_daily, context_dim), dtype=np.float32)
        
    # Targets: pickup probabilities p_T in [0, 1] modulated by supply v_r and demand o_r
    # p_r = o_r / (v_r + 1)
    target_demands = hourly_data[:, -1, :, 1]  # orders
    target_supplies = hourly_data[:, -1, :, 0] # vacant taxis
    targets = np.clip(target_demands / (target_supplies + 1.0 + 1e-4) * base_probs, 0.01, 0.95).astype(np.float32)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'hourly_features': hourly_data,
            'daily_features': daily_data,
            'hourly_context': hourly_ctx,
            'daily_context': daily_ctx
        },
        targets
    )).batch(batch_size)
    
    print("\n" + "-" * 70)
    print("  Starting Model Training (5 Epochs)...")
    print("-" * 70)
    history = model.fit(train_dataset, epochs=5, verbose=1)
    
    # Evaluation with weighted MAE (wMAE)
    preds = model.predict({
        'hourly_features': hourly_data,
        'daily_features': daily_data,
        'hourly_context': hourly_ctx,
        'daily_context': daily_ctx
    })
    
    trip_weights = np.sum(target_demands, axis=-1)
    w_mae = calculate_weighted_mae(targets, preds, weights=trip_weights)
    print(f"\nFinal Evaluation:")
    print(f"  - Training Loss (MAE): {history.history['loss'][-1]:.4f}")
    print(f"  - Weighted MAE (wMAE): {w_mae:.4f}")
    
    os.makedirs('results', exist_ok=True)
    weights_path = os.path.join('results', 'gcn_lstm_weights.weights.h5')
    model.save_weights(weights_path)
    print(f"\n[Success] Model weights saved to: {weights_path}")
    print("=" * 70)

if __name__ == '__main__':
    train()
