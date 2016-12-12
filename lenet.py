"""
LeNet Architecture

HINTS for layers:

    Convolutional layers:

    tf.nn.conv2d
    tf.nn.max_pool

    For preparing the convolutional layer output for the
    fully connected layers.

    tf.contrib.flatten
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import flatten

# Global Control
DEBUG = False

# Constants
EPOCHS = 10
BATCH_SIZE = 50
LEARNING_RATE = 0.001

n_input = 784  # MNIST data input (Shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Initial Weights

layer_width = {
    'l1': 6,
    'l2': 16,
    'l3': 120,
    'fc': n_classes
    }


weights = {
    'l1': tf.Variable(
            tf.truncated_normal(
                [5, 5, 1, layer_width['l1']])), 
    'l2': tf.Variable(
            tf.truncated_normal(
                [5, 5, layer_width['l1'], layer_width['l2']])), 
    'l3': tf.Variable(
            tf.truncated_normal(
                [5, 5, layer_width['l2'], layer_width['l3']])), 
    }

biases = {
    'l1': tf.Variable(tf.zeros(layer_width['l1'])),
    'l2': tf.Variable(tf.zeros(layer_width['l2'])),
    'l3': tf.Variable(tf.zeros(layer_width['l3'])),
    'out': tf.Variable(tf.zeros(n_classes))
}


# LeNet
    #Convolution layer 1. The output shape should be 28x28x6.
    #Activation 1. Your choice of activation function.
    #Pooling layer 1. The output shape should be 14x14x6.
    # Solution: 
        # 5x5 Kernel gives 32->28. No Strides. 'VALID' Padding; 
        # Pooling halves the Size.
    
    #Convolution layer 2. The output shape should be 10x10x16.
    #Activation 2. Your choice of activation function.
    #Pooling layer 2. The output shape should be 5x5x16.
    # Solution: 
        # 5x5 Kernel gives 14->10. No Strides. 'VALID' Padding; 
        # Pooling halves the Size.
    
    #Flatten layer. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. 
    #    The easiest way to do is by using tf.contrib.layers.flatten, which is already imported for you.
    
    #Fully connected layer 1. This should have 120 outputs.
    #Activation 3. Your choice of activation function.
    
    #Fully connected layer 2. This should have 10 outputs.

# Utility Functions

def conv2d(x, W, b, strides=1, padding='VALID'):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.tanh(x)


def maxpool2d(x, k=2, padding='SAME'):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding=padding)


# LeNet architecture:
# INPUT -> CONV -> ACT -> POOL -> CONV -> ACT -> POOL -> FLATTEN -> FC -> ACT -> FC
#
# Don't worry about anything else in the file too much, all you have to do is
# create the LeNet and return the result of the last fully connected layer.
def LeNet(x):
    # Reshape from 2D to 4D. This prepares the data for
    # convolutional and pooling layers.
    x = tf.reshape(x, (-1, 28, 28, 1))
    # Pad 0s to 32x32. Centers the digit further.
    # Add 2 rows/columns on each side for height and width dimensions.
    x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="CONSTANT")
    
    # TODO: Define the LeNet architecture.
    
    conv1 = conv2d(x, weights['l1'], biases['l1'])
    conv1 = maxpool2d(conv1)

    conv2 = conv2d(conv1, weights['l2'], biases['l2'])
    conv2 = maxpool2d(conv2)

    conv3 = conv2d(conv2, weights['l3'], biases['l3'])
    # conv3 = maxpool2d(conv3)
    
    flat = tf.contrib.layers.flatten(conv3)
    #    flat = maxpool2d(flat, k=1)
    # TODO: Do I need an activiation function here?
    
    fc = tf.contrib.layers.fully_connected(flat, layer_width['fc'])
    
    # Return the result of the last fully connected layer.
    return fc


# MNIST consists of 28x28x1, grayscale images
x = tf.placeholder(tf.float32, (None, n_input))
# Classify over 10 digits 0-9
y = tf.placeholder(tf.float32, (None, n_classes))
fc2 = LeNet(x)

if not DEBUG:
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc2, y))
    opt = tf.train.AdamOptimizer()
    train_op = opt.minimize(loss_op)
    correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y, 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def eval_data(dataset):
    """
    Given a dataset as input returns the loss and accuracy.
    """
    # If dataset.num_examples is not divisible by BATCH_SIZE
    # the remainder will be discarded.
    # Ex: If BATCH_SIZE is 64 and training set has 55000 examples
    # steps_per_epoch = 55000 // 64 = 859
    # num_examples = 859 * 64 = 54976
    #
    # So in that case we go over 54976 examples instead of 55000.
    steps_per_epoch = dataset.num_examples // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    total_acc, total_loss = 0, 0
    for step in range(steps_per_epoch):
        batch_x, batch_y = dataset.next_batch(BATCH_SIZE)
        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={x: batch_x, y: batch_y})
        total_acc += (acc * batch_x.shape[0])
        total_loss += (loss * batch_x.shape[0])

        if DEBUG: 
            print("DEBUG: Break eval_data()")
            break;

    return total_loss/num_examples, total_acc/num_examples

    
def main():
    # Load data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    
#    with tf.session() as debugSession:
#        debugSession.run(tf.conv1)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        steps_per_epoch = mnist.train.num_examples // BATCH_SIZE
        num_examples = steps_per_epoch * BATCH_SIZE

        # Train model
        for i in range(EPOCHS):
            for step in range(steps_per_epoch):
                batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
                
                if DEBUG:
                    conv = sess.run(fc2, feed_dict={x: batch_x, y: batch_y})
                    print (conv.shape)
                    print("DEBUG: Break steps_per_epoch")
                    break
                    
                else:
                    loss = sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

#            if DEBUG: 
#                print("DEBUG: return main()")
#                return

            val_loss, val_acc = eval_data(mnist.validation)
            print("EPOCH {} ...".format(i+1))
            print("Validation loss = {:.3f}".format(val_loss))
            print("Validation accuracy = {:.3f}".format(val_acc))
            print()
            
            
        # Evaluate on the test data
        test_loss, test_acc = eval_data(mnist.test)
        print("Test loss = {:.3f}".format(test_loss))
        print("Test accuracy = {:.3f}".format(test_acc))


if __name__ == '__main__':
    main()
