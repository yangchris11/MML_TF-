from __future__ import print_function

import sys
import time 
import argparse

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

import tensorflow as tf

learning_rate = 0.5
training_epochs = 15 
batch_size = 100 
display_step = 1 

n_hidden_layer = 1024

x = tf.placeholder( "float" , [ None , 784 ] )
y = tf.placeholder( "float" , [ None , 10 ] )

def multilayer_perceptron( x , W , B ):
    layer1 = tf.add( tf.matmul(x,W['h1']),B['b1'] )
    layer1 = tf.nn.relu(layer1)
    out_layer = tf.matmul(layer1,W['out'])+B['out']
    return out_layer

W = {
    'h1': tf.Variable(tf.random_normal([ 784 , n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_hidden_layer, 10]))
}

B = {
    'b1': tf.Variable(tf.random_normal([n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([10]))
}

pred = multilayer_perceptron(x,W,B)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,y: mnist.test.labels}))
