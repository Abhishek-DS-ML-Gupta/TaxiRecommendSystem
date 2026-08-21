import os
import pandas as pd
import numpy as np
import scipy.sparse as sparse
import networkx as nx

class DataProcessor:
    """
    Data processor for loading road network topology, segment-level properties,
    meteorology (weather) context features, and constructing graph matrices for GCN-LSTM.
    """
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir

    def load_network(self):
        """
        Loads the road network from edges and vertices.
        """
        edges_path = os.path.join(self.data_dir, 'refined_edges.csv')
        vertices_path = os.path.join(self.data_dir, 'refined_vertice.csv')
        
        edges_df = pd.read_csv(edges_path, header=None, names=['edge_id', 'source_id', 'target_id'])
        vertices_df = pd.read_csv(vertices_path, header=None, names=['node_id', 'lat', 'lon'])
        
        G = nx.DiGraph()
        for _, row in vertices_df.iterrows():
            G.add_node(int(row['node_id']), lat=row['lat'], lon=row['lon'])
            
        for _, row in edges_df.iterrows():
            G.add_edge(int(row['source_id']), int(row['target_id']), edge_id=int(row['edge_id']))
            
        return G, edges_df, vertices_df

    def load_network_properties(self):
        """
        Loads calculated road segment properties from network_properties.csv.
        Columns:
        [edge_id, target_edge_id, start_lat, start_lon, end_lat, end_lon, 
         vacant_taxis, pickup_events, pickup_prob, expected_earning]
        """
        path = os.path.join(self.data_dir, 'network_properties.csv')
        if not os.path.exists(path):
            return None
            
        col_names = [
            'edge_id', 'target_edge_id', 'start_lat', 'start_lon', 
            'end_lat', 'end_lon', 'vacant_taxis', 'pickup_events', 
            'pickup_prob', 'expected_earning'
        ]
        df = pd.read_csv(path, header=None, names=col_names)
        return df

    def load_weather_features(self):
        """
        Loads and standardizes hourly meteorology context features from weather.csv.
        Features considered per paper: temperature, dewPoint, humidity, windSpeed, 
        pressure, precipProbability, uvIndex, visibility.
        """
        path = os.path.join(self.data_dir, 'weather.csv')
        if not os.path.exists(path):
            return None, []
            
        df = pd.read_csv(path)
        feature_cols = [
            'temperature', 'dewPoint', 'humidity', 'windSpeed', 
            'pressure', 'precipProbability', 'uvIndex', 'visibility'
        ]
        
        # Fill missing columns/values with column medians or zeros
        for col in feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median() if not df[col].isna().all() else 0.0)
            else:
                df[col] = 0.0
                
        features = df[feature_cols].values.astype(np.float32)
        # Normalize features (z-score)
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True) + 1e-6
        normalized_features = (features - mean) / std
        return normalized_features, feature_cols

    def load_sample_input(self):
        """
        Loads the sample input for the prediction model as provided in the repository.
        Returns a 3D numpy array.
        """
        path_A = os.path.join(self.data_dir, 'sample_input', 'sample_last_dim_0.npz')
        path_B = os.path.join(self.data_dir, 'sample_input', 'sample_last_dim_1.npz')
        path_C = os.path.join(self.data_dir, 'sample_input', 'sample_last_dim_2.npz')
        
        if not (os.path.exists(path_A) and os.path.exists(path_B) and os.path.exists(path_C)):
            return None
            
        sA = sparse.load_npz(path_A)
        sB = sparse.load_npz(path_B)
        sC = sparse.load_npz(path_C)

        a = sA.toarray()
        b = sB.toarray()
        c = sC.toarray()

        m = np.concatenate((np.expand_dims(a, -1), np.expand_dims(b, -1), np.expand_dims(c, -1)), axis=-1)
        return m

    def construct_adjacency_matrix(self, edges_df):
        """
        Fast vectorized construction of road segment adjacency matrix.
        Two road segments (edges) are connected if the target node of edge 1 is the source node of edge 2.
        """
        edge_ids = edges_df['edge_id'].unique()
        edge_to_idx = {edge_id: idx for idx, edge_id in enumerate(edge_ids)}
        N = len(edge_ids)
        
        # Merge edges on target_id == source_id (vectorized in C/pandas)
        transitions = edges_df[['edge_id', 'target_id']].merge(
            edges_df[['edge_id', 'source_id']],
            left_on='target_id',
            right_on='source_id',
            suffixes=('_from', '_to')
        )
        
        from_indices = transitions['edge_id_from'].map(edge_to_idx).values
        to_indices = transitions['edge_id_to'].map(edge_to_idx).values
        data = np.ones(len(transitions), dtype=np.float32)
        
        A = sparse.coo_matrix((data, (from_indices, to_indices)), shape=(N, N))
        return A, edge_to_idx

    def calculate_normalized_laplacian(self, A):
        """
        Calculates the normalized Laplacian used in GCN: D^(-1/2) * A_tilde * D^(-1/2)
        where A_tilde = A + I_N.
        """
        N = A.shape[0]
        A_tilde = A + sparse.eye(N, dtype=np.float32)
        
        # Degree matrix
        D_tilde_diag = np.array(A_tilde.sum(axis=1)).flatten()
        
        # D^(-1/2)
        with np.errstate(divide='ignore'):
            D_tilde_inv_sqrt_diag = np.power(D_tilde_diag, -0.5)
        D_tilde_inv_sqrt_diag[np.isinf(D_tilde_inv_sqrt_diag)] = 0.
        D_tilde_inv_sqrt_diag[np.isnan(D_tilde_inv_sqrt_diag)] = 0.
        
        D_tilde_inv_sqrt = sparse.diags(D_tilde_inv_sqrt_diag, dtype=np.float32)
        
        return D_tilde_inv_sqrt.dot(A_tilde).dot(D_tilde_inv_sqrt).tocsr()

    def construct_segment_graph(self, edges_df, network_props_df=None):
        """
        Constructs a directed graph where nodes are road segments (edges in road network)
        and edges are valid driving transitions between consecutive road segments.
        Attaches pickup probabilities, expected earnings, travel times, and segment lengths.
        """
        route_G = nx.DiGraph()
        
        # Load edge IDs
        edge_ids = edges_df['edge_id'].unique()
        for edge_id in edge_ids:
            route_G.add_node(edge_id)
            
        # Fast transitions using merge
        transitions = edges_df[['edge_id', 'target_id']].merge(
            edges_df[['edge_id', 'source_id']],
            left_on='target_id',
            right_on='source_id',
            suffixes=('_from', '_to')
        )
        
        for _, row in transitions.iterrows():
            from_edge = int(row['edge_id_from'])
            to_edge = int(row['edge_id_to'])
            route_G.add_edge(from_edge, to_edge)
            
        # Attach properties if available
        pickup_probs = {}
        earnings = {}
        travel_times = {}
        segment_lengths = {}
        vacant_taxis = {}
        pickup_orders = {}
        
        if network_props_df is not None:
            # Aggregate properties by edge_id
            grouped = network_props_df.groupby('edge_id').first().reset_index()
            for _, row in grouped.iterrows():
                eid = int(row['edge_id'])
                pickup_probs[eid] = float(row['pickup_prob'])
                earnings[eid] = float(row['expected_earning'])
                vacant_taxis[eid] = float(row['vacant_taxis'])
                pickup_orders[eid] = float(row['pickup_events'])
                
                # Approximate segment length from coordinates (Haversine approx in meters)
                lat1, lon1 = float(row['start_lat']), float(row['start_lon'])
                lat2, lon2 = float(row['end_lat']), float(row['end_lon'])
                dlat = (lat2 - lat1) * 111000
                dlon = (lon2 - lon1) * 85000
                length_m = max(50.0, np.sqrt(dlat**2 + dlon**2))
                segment_lengths[eid] = length_m
                
                # Travel time at 20 km/h = 5.56 m/s (in minutes) per Definition 1
                travel_times[eid] = max(0.2, (length_m / 5.56) / 60.0)
        else:
            # Fallback default values
            for eid in edge_ids:
                pickup_probs[eid] = 0.25
                earnings[eid] = 25.0
                travel_times[eid] = 1.0
                segment_lengths[eid] = 300.0
                vacant_taxis[eid] = 5.0
                pickup_orders[eid] = 2.0
                
        return route_G, pickup_probs, earnings, travel_times, segment_lengths, vacant_taxis, pickup_orders

    def extract_virtual_stations(self, network_props_df, top_k=90):
        """
        Extracts candidate virtual stations (Definition 5, Figure 1(b)).
        Criteria: Top hot spots / taxi stands with high pickup events and profitable earnings.
        """
        if network_props_df is None:
            return {}
            
        grouped = network_props_df.groupby('edge_id').agg({
            'pickup_events': 'max',
            'expected_earning': 'mean',
            'vacant_taxis': 'mean',
            'start_lat': 'first',
            'start_lon': 'first'
        }).reset_index()
        
        # Filter high demand segments
        hot_spots = grouped.sort_values(by=['pickup_events', 'expected_earning'], ascending=False).head(top_k)
        
        virtual_stations = {}
        for _, row in hot_spots.iterrows():
            eid = int(row['edge_id'])
            # Waiting time in minutes (typically 3-8 minutes during peak hours per paper)
            waiting_time = max(3.0, min(10.0, float(row['vacant_taxis']) * 0.5))
            earning = max(20.0, float(row['expected_earning']))
            virtual_stations[eid] = {
                'edge_id': eid,
                'waiting_time': waiting_time,
                'expected_earning': earning,
                'lat': float(row['start_lat']),
                'lon': float(row['start_lon'])
            }
            
        return virtual_stations
