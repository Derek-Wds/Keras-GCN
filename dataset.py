from __future__ import print_function

import scipy.sparse as sp
import numpy as np
import tensorflow as tf

class CoraData():
    def __init__(self, data_root="data/cora/"):

        self._data_root = data_root

        self._data = self.process_data()


    def load_data(self, dataset="cora"):

        print('Loading {} dataset...'.format(dataset))

        idx_features_labels = np.genfromtxt("{}{}.content".format(self._data_root, dataset), dtype=np.dtype(str))

        edges = np.genfromtxt("{}{}.cites".format(self._data_root, dataset), dtype=np.int32)

        return idx_features_labels, edges

    def process_data(self):
        
        print("Process data ...")

        idx_features_labels, edges = self.load_data()

        features = idx_features_labels[:, 1:-1].astype(np.float32)
        features = self.normalize_feature(features)

        y = idx_features_labels[:, -1]
        labels = self.encode_onehot(y)

        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edge_indexs = np.array(list(map(idx_map.get, edges.flatten())), dtype=np.int32)
        edge_indexs = edge_indexs.reshape(edges.shape)
        adjacency = sp.coo_matrix((np.ones(len(edge_indexs)),
                    (edge_indexs[:, 0], edge_indexs[:, 1])),
                    shape=(features.shape[0], features.shape[0]), dtype="float32")

        adjacency = adjacency + adjacency.T.multiply(adjacency.T > adjacency) - adjacency.multiply(adjacency.T > adjacency)
        adjacency = self.normalize_adj(adjacency, symmetric=True)

        nf_shape = features.data.shape
        na_shape = adjacency.data.shape
        features = tf.SparseTensor(
                        indices=np.array(list(zip(features.row, features.col)), dtype=np.int64),
                        values=tf.cast(features.data, tf.float32),
                        dense_shape=features.shape)
        adjacency = tf.SparseTensor(
                        indices=np.array(list(zip(adjacency.row, adjacency.col)), dtype=np.int64),
                        values=tf.cast(adjacency.data, tf.float32),
                        dense_shape=adjacency.shape)

        train_index = np.arange(140)
        val_index = np.arange(200, 500)
        test_index = np.arange(500, 1500)

        train_mask = np.zeros(edge_indexs.shape[0], dtype = np.bool)
        val_mask = np.zeros(edge_indexs.shape[0], dtype = np.bool)
        test_mask = np.zeros(edge_indexs.shape[0], dtype = np.bool)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True

        print('Dataset has {} nodes, {} edges, {} features.'.format(features.shape[0], adjacency.shape[0], features.shape[1]))

        return features, labels, adjacency, train_mask, val_mask, test_mask, nf_shape, na_shape

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return labels_onehot

    def normalize_adj(self, adjacency, symmetric=True):
        adjacency += sp.eye(adjacency.shape[0])

        if symmetric:
            """L=D^-0.5 * (A+I) * D^-0.5"""
            d = sp.diags(np.power(np.array(adjacency.sum(1)), -0.5).flatten(), 0)
            a_norm = adjacency.dot(d).transpose().dot(d).tocoo()
        else:
            """L=D^-1 * (A+I)"""
            d = sp.diags(np.power(np.array(adjacency.sum(1)), -1).flatten(), 0)
            a_norm = d.dot(adjacency).tocoo()

        return a_norm

    def normalize_feature(self, features):

        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return sp.csr_matrix(features).tocoo()


    def data(self):
        return self._data