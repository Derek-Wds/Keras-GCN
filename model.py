# https://github.com/tkipf/keras-gcn
import tensorflow as tf
from tensorflow.keras import activations, initializers, constraints
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from utils import sparse_dropout

# Graph Convolutional Layer
class GCNLayer(tf.keras.layers.Layer):
 
    def __init__(self, units, use_bias=True, sparse_input=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GCNLayer, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.sparse_input = sparse_input
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
    
    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        self.built = True
    
    def call(self, input, adj):
        assert isinstance(adj, tf.SparseTensor), "Adjacency matrix should be a SparseTensor"
        if self.sparse_input:
            support = tf.sparse.sparse_dense_matmul(input, self.kernel)
        else:
            support = tf.matmul(input, self.kernel)
        output = tf.sparse.sparse_dense_matmul(adj, support)
        if self.use_bias:
            output = output + self.bias
        else:
            output = output
        return output

    def get_config(self):
        config = {'units': self.units,
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(GCNLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Graph Convolutional Neural Networks
class GCN(tf.keras.Model):
    def __init__(self, nhid, nclass, dropout, n_shape):
        super(GCN, self).__init__()

        self.gc1 = GCNLayer(nhid, kernel_regularizer=tf.keras.regularizers.l2(5e-4), sparse_input=True)
        self.gc2 = GCNLayer(nclass)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.n_shape = n_shape

    def call(self, x, adj, training=True):
        # x = sparse_dropout(x, 0.5, self.n_shape)
        x = tf.keras.activations.relu(self.gc1(x, adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)

        return tf.keras.activations.softmax(x)
        # return x
    
    def predict(self):
        return tf.keras.activations.softmax(self.outputs)

def optimizer(learning_rate=0.01):
    """Default optimizer name. Used in model.compile."""
    return tf.keras.optimizers.Adam(lr=learning_rate)

def prepare_prediction_column(prediction):
    """Return the prediction directly."""
    return prediction

def loss(labels, output):
    return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, output))