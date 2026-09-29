import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np

class GCNLayer(layers.Layer):
    """
    Graph Convolutional Network Layer using static SparseTensor for laplacian.
    Implements: G^l = sigma( D^{-1/2} * A_tilde * D^{-1/2} * G^{l-1} * Theta )
    """
    def __init__(self, output_dim, laplacian_tf, input_dim=None, activation='relu', **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.laplacian_tf = laplacian_tf
        self.input_dim = input_dim
        self.activation = tf.keras.activations.get(activation)
        if self.input_dim is not None:
            self.theta = self.add_weight(
                shape=(self.input_dim, self.output_dim),
                initializer='glorot_uniform',
                name='theta',
                trainable=True
            )

    def build(self, input_shape):
        if not hasattr(self, 'theta'):
            input_dim = input_shape[-1]
            if input_dim is None and self.input_dim is not None:
                input_dim = self.input_dim
            self.theta = self.add_weight(
                shape=(input_dim, self.output_dim),
                initializer='glorot_uniform',
                name='theta',
                trainable=True
            )
        super(GCNLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs: (batch_size, num_nodes, input_dim)
        xw = tf.matmul(inputs, self.theta)
        
        # xw is (batch, N, out_dim)
        batch_size = tf.shape(xw)[0]
        N = tf.shape(xw)[1]
        out_dim = tf.shape(xw)[2]
        
        # reshape for sparse matmul
        xw_reshaped = tf.reshape(tf.transpose(xw, [1, 0, 2]), (N, batch_size * out_dim))
        
        # (N, N) * (N, batch * out_dim) -> (N, batch * out_dim)
        out_reshaped = tf.sparse.sparse_dense_matmul(self.laplacian_tf, xw_reshaped)
        
        # reshape back to (batch, N, out_dim)
        output = tf.transpose(tf.reshape(out_reshaped, (N, batch_size, out_dim)), [1, 0, 2])
        
        if self.activation is not None:
            output = self.activation(output)
            
        return output

class GCN_LSTM_Model(tf.keras.Model):
    def __init__(self, num_nodes, laplacian_tf, num_features=2, gcn_units=32, lstm_units=16, output_dim=1, **kwargs):
        super(GCN_LSTM_Model, self).__init__(**kwargs)
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.gcn_units = gcn_units
        
        self.gcn = GCNLayer(output_dim=gcn_units, laplacian_tf=laplacian_tf, input_dim=num_features, activation='relu')
        self.lstm_hourly = layers.LSTM(lstm_units, return_sequences=False)
        self.lstm_daily = layers.LSTM(lstm_units, return_sequences=False)
        self.dense = layers.Dense(output_dim, activation='sigmoid')

    def call(self, inputs, training=False):
        hourly = inputs['hourly_features']
        daily = inputs['daily_features']
        
        batch_size = tf.shape(hourly)[0]
        seq_hourly = tf.shape(hourly)[1]
        
        feat_dim = hourly.shape[-1] if hourly.shape[-1] is not None else self.num_features
        hourly_reshaped = tf.reshape(hourly, (-1, self.num_nodes, feat_dim))
        gcn_hourly = self.gcn(hourly_reshaped)
        gcn_hourly = tf.reshape(gcn_hourly, (batch_size, seq_hourly, self.num_nodes * self.gcn_units))
        
        lstm_out_hourly = self.lstm_hourly(gcn_hourly)
        
        seq_daily = tf.shape(daily)[1]
        daily_reshaped = tf.reshape(daily, (-1, self.num_nodes, feat_dim))
        gcn_daily = self.gcn(daily_reshaped)
        gcn_daily = tf.reshape(gcn_daily, (batch_size, seq_daily, self.num_nodes * self.gcn_units))
        
        lstm_out_daily = self.lstm_daily(gcn_daily)
        
        concatenated = tf.concat([lstm_out_hourly, lstm_out_daily], axis=-1)
        out = self.dense(concatenated)
        return out

def build_model(num_nodes, seq_len_hourly, seq_len_daily, num_features, laplacian_tf, gcn_units=32, lstm_units=16):
    """
    Builds the Keras model.
    """
    model = GCN_LSTM_Model(
        num_nodes=num_nodes,
        laplacian_tf=laplacian_tf,
        num_features=num_features,
        gcn_units=gcn_units,
        lstm_units=lstm_units,
        output_dim=num_nodes
    )
    return model

if __name__ == '__main__':
    # Test model build
    num_nodes = 10
    seq_len_hourly = 3
    seq_len_daily = 3
    num_features = 2
    
    # Dummy laplacian sparse tensor
    indices = [[i, i] for i in range(num_nodes)]
    values = np.ones(num_nodes, dtype=np.float32)
    L_tf = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=[num_nodes, num_nodes])
    
    model = build_model(num_nodes, seq_len_hourly, seq_len_daily, num_features, L_tf)
    
    # Dummy data
    h_data = np.random.rand(4, seq_len_hourly, num_nodes, num_features).astype(np.float32)
    d_data = np.random.rand(4, seq_len_daily, num_nodes, num_features).astype(np.float32)
    
    inputs = {
        'hourly_features': h_data,
        'daily_features': d_data
    }
    
    output = model(inputs)
    print("Model output shape:", output.shape)
