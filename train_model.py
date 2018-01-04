import numpy as np
import tensorflow as tf


LR = 1e-4
IMAGE_SIZE = 128
CLASSES = 10
class_name = ['apple-pie', 'chocolate-cake', 'donuts', 'eggs-benedict', 'french-toast',
              'gnocchi', 'grilled-salmon', 'pancakes', 'shrimp-and-grits', 'spaghetti-bolognese']


MODEL_NAME = 'Food-image-classifer-{}-{}-model'.format(LR, '2conv')
MODEL_PATH = ''

def load_train():
    return np.load('train_set.npy')

def load_test():
    return np.load('test_set.npy')

train_set = load_train()
# train_set.shape

train_X = np.array([i[0] for i in train_set]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
train_y = np.array([list(i[1]) for i in train_set])

# print("Train_X shape: ", train_X.shape)
# print("Train_y shape: ", train_y.shape)

# Load train batch
def train_batch(size=16):
    first_index,last_index = 0, size
    while last_index <= len(train_X):
        # print('First Index: ', first_index)
        # print('Last Index: ', last_index)
        yield train_X[first_index: last_index], train_y[first_index: last_index]
        first_index, last_index = last_index, last_index + size


# Testing Datasets
test_set = load_test()
# test_set.shape

test_X = np.array([i[0] for i in test_set]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
test_y = np.array([list(i[1]) for i in test_set])

# print("Train_X shape: ", train_X.shape)
# print("Train_y shape: ", train_y.shape)


# Place Holder
X = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name='X')
y_true = tf.placeholder(tf.float32, shape=[None, CLASSES], name="y_true")
y_true_cls = tf.argmax(y_true, axis=1)


# Weight Initialization - Helper function

def weight_variable(name, shape):
    W = tf.get_variable(name, shape,
           initializer=tf.contrib.layers.xavier_initializer())
    return W

def bias_variable(shape):
    initial = tf.constant(0.5, shape=shape)
    return tf.Variable(initial)


# Convolution and Pooling - Helper Function

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding="SAME")


# CNN Model

# Frist Conv layer
W_conv1 = weight_variable('W_conv1', [5, 5, 3, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second Conv Layer
W_conv2 = weight_variable('W_conv2', [5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Third Conv Layer
W_conv3 = weight_variable('W_conv3', [5, 5, 64, 128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# print("h_pool3 shape: ", h_pool3.get_shape())

# Fully or Densely Connected Layer
W_fc1 = weight_variable('W_fc1', [16*16*128, 1024])
b_fc1 = bias_variable([1024])

h_pool3_flat = tf.reshape(h_pool3, [-1, 16*16*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

# print("h_fc1 shape: ", h_fc1.get_shape())

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
W_fc2 = weight_variable('W_fc2', [1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# print("y_conv shape: ", y_conv.get_shape())


# Training Model

# Feed forward - Prediction
y_pred = tf.nn.softmax(y_conv)
y_pred_cls = tf.argmax(y_pred, axis=1)

# Training Objective
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_true)
#cost = tf.reduce_mean(cross_entropy)
train_step = tf.train.AdamOptimizer(LR).minimize(cross_entropy)

# Evaluation
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        for batch in train_batch():
            train_step.run(feed_dict={X: batch[0], y_true: batch[1], keep_prob: 0.5})
        if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                    X: batch[0], y_true: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
    test_accuracy = accuracy.eval(feed_dict={X: test_X, y_true: test_y, keep_prob: 1.0})
    print("Test Accuracy: %g", test_accuracy)
    MODEL_PATH = saver.save(sess, 'model/' + MODEL_NAME)
