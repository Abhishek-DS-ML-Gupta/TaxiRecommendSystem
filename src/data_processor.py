import pandas as pd
import numpy as np
import scipy.sparse as sparse
import os
import networkx as nx

class DataProcessor:
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

    def load_sample_input(self):
        """
        Loads the sample input for the prediction model as provided in the repository.
        Returns a 3D numpy array.
        """
        path_A = os.path.join(self.data_dir, 'sample_input', 'sample_last_dim_0.npz')
        path_B = os.path.join(self.data_dir, 'sample_input', 'sample_last_dim_1.npz')
        path_C = os.path.join(self.data_dir, 'sample_input', 'sample_last_dim_2.npz')
        
        if not (os.path.exists(path_A) and os.path.exists(path_B) and os.path.exists(path_C)):
            print("Sample input files not found. Please ensure they are present in data/sample_input/")
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
        Constructs the adjacency matrix for road segments (edges) using sparse matrices.
        """
        edge_ids = edges_df['edge_id'].unique()
        edge_to_idx = {edge_id: idx for idx, edge_id in enumerate(edge_ids)}
        N = len(edge_ids)
        
        # Build adjacency matrix A as sparse
        rows, cols, data = [], [], []
        
        source_group = edges_df.groupby('source_id')
        
        for _, row in edges_df.iterrows():
            target_node = row['target_id']
            edge_id_1 = row['edge_id']
            idx_1 = edge_to_idx[edge_id_1]
            
            if target_node in source_group.groups:
                next_edges = source_group.get_group(target_node)
                for _, next_row in next_edges.iterrows():
                    edge_id_2 = next_row['edge_id']
                    idx_2 = edge_to_idx[edge_id_2]
                    rows.append(idx_1)
                    cols.append(idx_2)
                    data.append(1.0)
                    
        A = sparse.coo_matrix((data, (rows, cols)), shape=(N, N))
        return A, edge_to_idx

    def calculate_normalized_laplacian(self, A):
        """
        Calculates the normalized laplacian used in GCN: D^(-1/2) * A_tilde * D^(-1/2)
        using sparse matrices.
        """
        N = A.shape[0]
        A_tilde = A + sparse.eye(N)
        
        # Degree matrix
        D_tilde_diag = np.array(A_tilde.sum(axis=1)).flatten()
        
        # D^(-1/2)
        D_tilde_inv_sqrt_diag = np.power(D_tilde_diag, -0.5)
        D_tilde_inv_sqrt_diag[np.isinf(D_tilde_inv_sqrt_diag)] = 0.
        
        D_tilde_inv_sqrt = sparse.diags(D_tilde_inv_sqrt_diag)
        
        return D_tilde_inv_sqrt.dot(A_tilde).dot(D_tilde_inv_sqrt)

if __name__ == '__main__':
    processor = DataProcessor(data_dir='../data')
    # Test loading network
    G, edges, vertices = processor.load_network()
    print(f"Loaded network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    sample_input = processor.load_sample_input()
    if sample_input is not None:
        print(f"Loaded sample input of shape {sample_input.shape}")
