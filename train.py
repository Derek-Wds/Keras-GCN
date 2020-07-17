from model import GCN
from dataset import CoraData
from utils import *
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from metrics import *

dataset = CoraData()
# data loading method 1
# features, labels, adj, train_mask, val_mask, test_mask, nf_shape, na_shape = dataset.data()

# data loading method 2
adj, features, labels, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')
support = [preprocess_adj(adj)]
adj = tf.cast(tf.SparseTensor(*support[0]), tf.float32)
features = preprocess_features(features).tocoo()
nf_shape = features.data.shape
features = tf.SparseTensor(
                indices=np.array(list(zip(features.row, features.col)), dtype=np.int64),
                values=tf.cast(features.data, tf.float32),
                dense_shape=features.shape)


graph = [features, adj]

model = GCN(16, labels.shape[-1], 0.5, nf_shape)

criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

def loss(model, x, y, train_mask, training=True):

    y_ = model(x[0], x[1], training=training)

    test_mask_logits = tf.gather_nd(y_, tf.where(train_mask))
    masked_labels = tf.gather_nd(y, tf.where(train_mask))

    return criterion(y_true=masked_labels, y_pred=test_mask_logits)
    # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=test_mask_logits, labels=masked_labels))
    # return masked_softmax_cross_entropy(y_, y, train_mask)

def grad(model, inputs, targets, train_mask):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, train_mask)
    
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def test(mask):
    logits = model(graph[0], graph[1], training=False)

    test_mask_logits = tf.gather_nd(logits, tf.where(mask))
    masked_labels = tf.gather_nd(labels, tf.where(mask))

    ll = tf.equal(tf.argmax(masked_labels, -1), tf.argmax(test_mask_logits, -1))
    accuarcy = tf.reduce_mean(tf.cast(ll, dtype=tf.float32))

    return accuarcy
    # return masked_accuracy(logits, labels, mask)

optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)

# 记录过程值，以便最后可视化
train_loss_results = []
train_accuracy_results = []
train_val_results = []

num_epochs = 200

for epoch in range(num_epochs):

    loss_value, grads = grad(model, graph, labels, train_mask)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    accuarcy = test(train_mask)
    val_acc = test(val_mask)

    train_loss_results.append(loss_value)
    train_accuracy_results.append(accuarcy)
    train_val_results.append(val_acc)

    print("Epoch {} loss={} accuracy={} val_acc={}".format(epoch, loss_value, accuarcy, val_acc))

test_acc = test(test_mask)
print("test-acc={}".format(test_acc))

# 训练过程可视化
fig, axes = plt.subplots(3, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].plot(train_accuracy_results)

axes[2].set_ylabel("Val Acc", fontsize=14)
axes[2].plot(train_val_results)

plt.show()
