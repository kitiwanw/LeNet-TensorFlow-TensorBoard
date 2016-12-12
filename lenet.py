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
douche = False

LOGDIR = "./logs/train/run14-cmdline-NOreset-tiny"

# Constants
EPOCHS = 2
#EPOCHS = 10
BATCH_SIZE = 50

n_input = 784  # MNIST data input (Shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Initial Weights

layer_width = {
    'l1': 6,
    'l2': 16,
    'fc1': 400,
#    'l3': 120,
    'fc2': 120,
    'out': n_classes
    }


weights = {
    'l1': tf.Variable(
            tf.truncated_normal(
                [5, 5, 1, layer_width['l1']]),
                name='Weights_l1'), 
    'l2': tf.Variable(
            tf.truncated_normal(
                [5, 5, layer_width['l1'], layer_width['l2']]),
                name='Weights_l2'),
#    'l3': tf.Variable(
#            tf.truncated_normal(
#                [5, 5, layer_width['l2'], layer_width['fc1']])), 
#    'fc1': tf.Variable(
#            tf.truncated_normal(
#                [5, 5, layer_width['l2'], layer_width['fc1']])), 
#    'fc2': tf.Variable(
#            tf.truncated_normal(
#                [5, 5, layer_width['fc1'], layer_width['fc2']])), 
#                name='Weights_fc2'),
#    'out': tf.Variable(
#            tf.truncated_normal(
#                [5, 5, layer_width['fc2'], layer_width['out']])), 
#                name='Weights_out'),
    }

biases = {
    'l1': tf.Variable(tf.zeros(layer_width['l1']), name='bias_l1'),
    'l2': tf.Variable(tf.zeros(layer_width['l2']), name='bias_l2'),
#    'l3': tf.Variable(tf.zeros(layer_width['l3'])),
    'fc1': tf.Variable(tf.zeros(layer_width['fc1']), name='bias_fc1'),
    'fc2': tf.Variable(tf.zeros(layer_width['fc2']), name='bias_fc2'),
    'out': tf.Variable(tf.zeros(layer_width['out']), name='bias_out')
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

def conv2d(name, x, W, b, strides=1, padding='VALID'):
    with tf.name_scope(name):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
        x = tf.nn.bias_add(x, b)
    #    return tf.nn.relu(x)
        return tf.nn.tanh(x)


def linear2d(x, W, b):
    x = tf.matmul(x, W) + b
#    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
    

def maxpool2d(name, x, k=2, padding='SAME'):
    with tf.name_scope(name):
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
    
    conv1 = conv2d('l1', x, weights['l1'], biases['l1'])
    conv1 = maxpool2d('l1', conv1)

    conv2 = conv2d('l2', conv1, weights['l2'], biases['l2'])
    conv2 = maxpool2d('l2', conv2)

    if douche:
        # Flatten
        fc1 = flatten(conv2)
        # (5 * 5 * 16, 120)
        fc1_shape = (fc1.get_shape().as_list()[-1], 120)
    
        fc1_W = tf.Variable(tf.truncated_normal(shape=(fc1_shape)))
        fc1_b = tf.Variable(tf.zeros(120))
        fc1 = tf.matmul(fc1, fc1_W) + fc1_b
        fc1 = tf.nn.relu(fc1)
    
        fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 10)))
        fc2_b = tf.Variable(tf.zeros(10))
        
#        return fc1_shape 
        return tf.matmul(fc1, fc2_W) + fc2_b
    
    else:
#        conv3 = conv2d(conv2, weights['l3'], biases['l3'])
        # conv3 = maxpool2d(conv3)

        with tf.name_scope('fc1'):
            fc1 = tf.contrib.layers.flatten(conv2)
            fc1_shape = (fc1.get_shape().as_list()[-1], 120)
            print ("fc1_shape:", fc1.get_shape(), fc1_shape)
            #    flat = maxpool2d(flat, k=1)
            # TODO: Do I need an activiation function here?
        
        with tf.name_scope('fc2'):
            fc2 = tf.contrib.layers.fully_connected(
                    inputs=fc1,
                    num_outputs=layer_width['fc2']) 
    #                biases_initializer=biases['fc2'])

        with tf.name_scope('out'):
            out = tf.contrib.layers.fully_connected(
                    inputs=fc2,
                    num_outputs=layer_width['out']) 
    #                biases_initializer=biases['out'])
        
        # Return the result of the last fully connected layer.
        return out 


print ("Globals: DEBUG(%s), Douche(%s)" % (DEBUG, douche))

# tf.reset_default_graph()

# MNIST consists of 28x28x1, grayscale images
x = tf.placeholder(tf.float32, (None, n_input))
# Classify over 10 digits 0-9
y = tf.placeholder(tf.float32, (None, n_classes))
fc2 = LeNet(x)

# Tensorboard Summary

# Histogram Summary: Weights
tf.histogram_summary("Weights_l1_summary", weights['l1'])
tf.histogram_summary("Weights_l2_summary", weights['l2'])

# Histogram Summary: Biases
tf.histogram_summary("Biases L1", biases['l1'])
tf.histogram_summary("Biases L2", biases['l2'])


if not DEBUG:
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc2, y))
    opt = tf.train.AdamOptimizer()
    train_op = opt.minimize(loss_op)
    correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y, 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Scalar Summary: Cost and Accuracy
    tf.scalar_summary("cost", loss_op)
    tf.scalar_summary("accuracy", accuracy_op)

def eval_data(sess, dataset, summary_op):
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

        loss, acc, summary = sess.run([loss_op, accuracy_op, summary_op], 
                                      feed_dict={x: batch_x, y: batch_y})
        total_acc += (acc * batch_x.shape[0])
        total_loss += (loss * batch_x.shape[0])

        if DEBUG: 
            print("DEBUG: Break eval_data()")
            break;

    return summary, total_loss/num_examples, total_acc/num_examples

    
def main():
    # Load data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    merged_summary_op = tf.merge_all_summaries()

    with tf.Session() as sess:
        # TensorBoard: Summary Writer

        writer = tf.train.SummaryWriter(LOGDIR, sess.graph)
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

            if DEBUG: 
                print("DEBUG: return main()")
                return

            val_summary, val_loss, val_acc = eval_data(sess, mnist.validation, merged_summary_op)
            print("EPOCH {} ...".format(i+1))
            print("Validation loss = {:.3f}".format(val_loss))
            print("Validation accuracy = {:.3f}".format(val_acc))
            print()

            # TensorBoard: Write Summaries
            writer.add_summary(val_summary, i*BATCH_SIZE)

        # Evaluate on the test data
        test_summary, test_loss, test_acc = eval_data(sess, mnist.test, merged_summary_op)
        print("Test loss = {:.3f}".format(test_loss))
        print("Test accuracy = {:.3f}".format(test_acc))


if __name__ == '__main__':
    main()
