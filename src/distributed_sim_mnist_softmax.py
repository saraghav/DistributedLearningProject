#!/usr/bin/python3 -i
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.python.framework import dtypes
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging

import ilogger
ilogger.setup_root_logger('/dev/null', logging.INFO)
logger = ilogger.setup_logger(__name__)

class MNISTSoftmaxRegression(object):
    
    def __init__(self, minibatch_size, learning_rate, n_epochs, mnist_train=None):
        self.minibatch_size = minibatch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        if mnist_train is None:
            self.load_mnist_data()
        else:
            self.mnist_train = mnist_train

        self.construct_model()
        self.train_model()

    def load_mnist_data(self):
        self.mnist_train = input_data.read_data_sets("MNIST_data/", one_hot=True).train

    def construct_model(self):
        # place holder for input with unknown number of examples
        self.x = tf.placeholder(tf.float32, [None, 784])
        
        # model parameters
        #  model: y_hat = softmax( W^T x + b )
        #  W.shape = n_features x n_classes
        self.W = tf.Variable(tf.random_normal([784, 10]))
        self.b = tf.Variable(tf.zeros([10]))

        # output
        self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)

        # labels and training
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        self.cross_entropy = tf.reduce_mean( -tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]) )
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy)
        
        # initialization
        self.init = tf.global_variables_initializer()

        # evaluation
        self.correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
        # initialize model
        if tf.get_default_session():
            logger.info('default session available; using default session for model')
            self.sess = tf.get_default_session()
        else:
            self.sess = tf.Session()
        self.sess.run(self.init)

    def train_model(self):
        for i in range(self.n_epochs):
            batch_xs, batch_ys = self.mnist_train.next_batch(self.minibatch_size)
            self.sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})

    def evaluate_model(self, test_data):
        accuracy_eval = self.sess.run(self.accuracy, feed_dict={self.x: test_data.images, self.y_: test_data.labels})
        return accuracy_eval


class DistSimulation(MNISTSoftmaxRegression):
    
    def __init__(self, n_machines, common_examples_fraction, *args):
        self.n_machines = n_machines
        self.common_examples_fraction = common_examples_fraction

        super().__init__(*args)

    def train_model(self):
        self.partition_data()
        self.train_distributed_models()
        self.combine_distributed_models()

    def partition_data(self):
        n_examples = self.mnist_train.images.shape[0]
        random_order = np.random.permutation(n_examples)
        
        n_common_examples = int(self.common_examples_fraction * n_examples)
        n_subset_examples = int( (n_examples-n_common_examples)/self.n_machines )

        common_examples_indices = random_order[0:n_common_examples]
        common_examples = self.mnist_train.images[common_examples_indices, :]
        common_examples_labels = self.mnist_train.labels[common_examples_indices, :]
        
        self.training_data_sets = []
        for i_machine in range(self.n_machines):
            slice_start = n_common_examples + n_subset_examples*i_machine
            slice_end = n_common_examples + n_subset_examples*(i_machine+1)
            subset_examples_indices = random_order[slice_start:slice_end]

            subset_examples = self.mnist_train.images[subset_examples_indices, :]
            subset_examples_labels = self.mnist_train.labels[subset_examples_indices, :]

            images = np.concatenate([common_examples, subset_examples], axis=0)
            labels = np.concatenate([common_examples_labels, subset_examples_labels], axis=0)
            # using dtype = dtypes.uint8 to prevent the DataSet class to scale the features by 1/255
            self.training_data_sets.append( DataSet(images, labels, reshape=False, dtype=dtypes.uint8) )

    def train_distributed_models(self):
        self.distributed_models = []
        for i_machine, mnist_train in enumerate(self.training_data_sets):
            model = MNISTSoftmaxRegression(self.minibatch_size, self.learning_rate, self.n_epochs, mnist_train)
            self.distributed_models.append(model)
    
    def combine_distributed_models(self):
        W_list = []
        b_list = []

        for i_machine, model in enumerate(self.distributed_models):
            W_list.append( model.W.eval(session=model.sess) )
            b_list.append( model.b.eval(session=model.sess) )

        W_avg = np.mean(W_list, axis=0)
        b_avg = np.mean(b_list, axis=0)
        assign_W = self.W.assign(W_avg)
        assign_b = self.b.assign(b_avg)
        self.sess.run(assign_W)
        self.sess.run(assign_b)
    
    def evaluate_distributed_models(self, test_data):
        accuracy_list = []
        for model in self.distributed_models:
            accuracy_list.append( model.evaluate_model(test_data) )
        return accuracy_list

if __name__=='__main__':
    minibatch_size = 100
    learning_rate = 0.5
    n_epochs = 1000

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    mnist_classifier = MNISTSoftmaxRegression(minibatch_size, learning_rate, n_epochs, mnist.train)
    accuracy = mnist_classifier.evaluate_model(mnist.test)

    print('accuracy = {0}'.format(accuracy))

    n_machines = 4
    common_examples_fraction = 1
    mnist_distributed = DistSimulation(n_machines, common_examples_fraction, minibatch_size, learning_rate, n_epochs, mnist.train)
    combined_model_accuracy = mnist_distributed.evaluate_model(mnist.test)
    dist_model_accuracy_list = mnist_distributed.evaluate_distributed_models(mnist.test)

    print('combined_model_accuracy = {0}'.format(combined_model_accuracy))
    print('dist_model_accuracy_list:')
    print(dist_model_accuracy_list)
