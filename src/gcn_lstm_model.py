import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class GCNLayer(layers.Layer):
    """
    Graph Convolutional Network Layer using static SparseTensor for normalized Laplacian.
    Implements: G^l = sigma( D_tilde^{-1/2} * A_tilde * D_tilde^{-1/2} * G^{l-1} * Theta )
    as specified in Equations 3 and 4 in Liu et al. (2023).
    """
    def __init__(self, output_dim, laplacian_tf, input_dim=None, activation='relu', **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.laplacian_tf = laplacian_tf
        self.activation = tf.keras.activations.get(activation)

        if self.input_dim is not None:
            self.theta = self.add_weight(
                shape=(self.input_dim, self.output_dim),
                initializer='glorot_uniform',
                name='theta',
                trainable=True
            )

    def build(self, input_shape):
        if not hasattr(self, 'theta') or self.theta is None:
            dim = input_shape[-1] if input_shape[-1] is not None else self.input_dim
            if dim is None:
                dim = 2 # default fallback
            self.theta = self.add_weight(
                shape=(dim, self.output_dim),
                initializer='glorot_uniform',
                name='theta',
                trainable=True
            )
        super(GCNLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, num_nodes, input_dim)
        xw = tf.matmul(inputs, self.theta)  # (batch_size, N, out_dim)
        
        batch_size = tf.shape(xw)[0]
        N = tf.shape(xw)[1]
        out_dim = self.output_dim
        
        # Reshape for sparse multiplication with static Laplacian (N, N)
        xw_reshaped = tf.reshape(tf.transpose(xw, [1, 0, 2]), (N, batch_size * out_dim))
        
        # (N, N) * (N, batch_size * out_dim) -> (N, batch_size * out_dim)
        out_reshaped = tf.sparse.sparse_dense_matmul(self.laplacian_tf, xw_reshaped)
        
        # Reshape back to (batch_size, N, out_dim)
        output = tf.transpose(tf.reshape(out_reshaped, (N, batch_size, out_dim)), [1, 0, 2])
        
        if self.activation is not None:
            output = self.activation(output)
            
        return output

class GCN_LSTM_Model(tf.keras.Model):
    """
    Integrated GCN-LSTM architecture for dynamic taxi pickup probability prediction (Section 4, Figure 5).
    Captures:
    - Spatial road network topological dependencies via GCN
    - Temporal dynamics via two-stream LSTM (hourly short-term and daily periodic long-term)
    - Meteorology / weather context features C_t via feature fusion: m_t = (C_t \oplus vec(G_t^L))
    """
    def __init__(self, num_nodes, num_features, laplacian_tf, gcn_units=32, lstm_units=16, 
                 output_dim=None, **kwargs):
        super(GCN_LSTM_Model, self).__init__(**kwargs)
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.output_dim = output_dim if output_dim is not None else num_nodes
        self.gcn_units = gcn_units
        
        self.gcn = GCNLayer(
            output_dim=gcn_units, 
            laplacian_tf=laplacian_tf, 
            input_dim=num_features, 
            activation='relu'
        )
        self.lstm_hourly = layers.LSTM(lstm_units, return_sequences=False)
        self.lstm_daily = layers.LSTM(lstm_units, return_sequences=False)
        self.dense = layers.Dense(self.output_dim, activation='sigmoid')

    def call(self, inputs, training=False):
        hourly = inputs['hourly_features']  # (batch, seq_h, num_nodes, num_features)
        daily = inputs['daily_features']    # (batch, seq_d, num_nodes, num_features)
        
        hourly_context = inputs.get('hourly_context', None) # (batch, seq_h, context_dim)
        daily_context = inputs.get('daily_context', None)   # (batch, seq_d, context_dim)
        
        batch_size = tf.shape(hourly)[0]
        seq_hourly = tf.shape(hourly)[1]
        
        # 1. Process hourly time series through GCN
        hourly_reshaped = tf.reshape(hourly, (batch_size * seq_hourly, self.num_nodes, self.num_features))
        gcn_hourly = self.gcn(hourly_reshaped)  # (batch * seq_h, N, gcn_units)
        gcn_hourly_flat = tf.reshape(gcn_hourly, (batch_size, seq_hourly, self.num_nodes * self.gcn_units))
        
        if hourly_context is not None:
            # Equation 5: m_t = (C_t \oplus vectorize(G_t^L))
            m_hourly = tf.concat([hourly_context, gcn_hourly_flat], axis=-1)
        else:
            m_hourly = gcn_hourly_flat
            
        lstm_out_hourly = self.lstm_hourly(m_hourly)
        
        # 2. Process daily periodic time series through GCN
        seq_daily = tf.shape(daily)[1]
        daily_reshaped = tf.reshape(daily, (batch_size * seq_daily, self.num_nodes, self.num_features))
        gcn_daily = self.gcn(daily_reshaped)  # (batch * seq_d, N, gcn_units)
        gcn_daily_flat = tf.reshape(gcn_daily, (batch_size, seq_daily, self.num_nodes * self.gcn_units))
        
        if daily_context is not None:
            m_daily = tf.concat([daily_context, gcn_daily_flat], axis=-1)
        else:
            m_daily = gcn_daily_flat
            
        lstm_out_daily = self.lstm_daily(m_daily)
        
        # 3. Concatenate temporal representations (h_{T-1} \oplus h_{T-24})
        concatenated = tf.concat([lstm_out_hourly, lstm_out_daily], axis=-1)
        
        # 4. Predict pickup probabilities p_T \in [0, 1]^N
        out = self.dense(concatenated)
        return out

def build_model(num_nodes, seq_len_hourly, seq_len_daily, num_features, laplacian_tf, 
                gcn_units=32, lstm_units=16, context_dim=0):
    """
    Builds and returns the GCN-LSTM model instance.
    """
    model = GCN_LSTM_Model(
        num_nodes=num_nodes, 
        num_features=num_features,
        laplacian_tf=laplacian_tf, 
        gcn_units=gcn_units, 
        lstm_units=lstm_units, 
        output_dim=num_nodes
    )
    return model

def calculate_weighted_mae(y_true, y_pred, weights=None):
    """
    Calculates the Weighted Mean Absolute Error (wMAE) as defined in Section 6.1.1:
    wMAE = sum(W(t) * MAE(t)) / sum(W(t))
    where W(t) is the total trip demand during period t.
    """
    mae_per_sample = np.mean(np.abs(y_true - y_pred), axis=-1)
    if weights is None:
        return float(np.mean(mae_per_sample))
    weights = np.asarray(weights, dtype=np.float32).flatten()
    return float(np.sum(weights * mae_per_sample) / (np.sum(weights) + 1e-8))
