#!/usr/bin/python3 -i
import input_data
from input_data import DataSet
from tensorflow.python.framework import dtypes
from os import path

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pdb
import logging

import ilogger
ilogger.setup_root_logger('/dev/null', logging.DEBUG)
logger = ilogger.setup_logger(__name__)

class MNISTSoftmaxRegression(object):
    
    def __init__(self, minibatch_size, learning_rate, n_iterations, mnist_train=None, model_name='classifier', write_summary=False, alpha=0.1):
        self.minibatch_size = minibatch_size
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.model_name = model_name
        # only ONE summary key is supported
        self.write_summary = write_summary
        self.summaries = ['/'.join([tf.GraphKeys.SUMMARIES, self.model_name])]

        self.alpha = alpha
        
        logger.info('data for {0} : images.shape = {1}, labels.shape = {2}'.format(model_name, mnist_train.images.shape, mnist_train.labels.shape))

        if mnist_train is None:
            self.load_mnist_data()
        else:
            self.mnist_train = mnist_train

        self.construct_model()

    def load_mnist_data(self):
        self.mnist_train = input_data.read_data_sets("MNIST_data/", one_hot=True).train

    def construct_model(self):
        def l2(weights):
            return tf.nn.l2_loss(weights)

        def l1(weights):
            return tf.reduce_sum( tf.abs(weights) )

        with tf.variable_scope(self.model_name):
            # place holder for input with unknown number of examples
            with tf.variable_scope('features'):
                self.x = tf.placeholder(tf.float32, [None, 784], name='x')
            
            # model parameters
            #  model: y_hat = softmax( W^T x + b )
            #  W.shape = n_features x n_classes
            #  also assign model parameters if needed
            with tf.variable_scope('model_parameters'):
                self.W = tf.Variable(tf.random_normal([784, 10]), name='W')
                # self.W = tf.Variable(tf.zeros([784, 10]), name='W')
                self.b = tf.Variable(tf.random_normal([10]), name='b')
                # self.b = tf.Variable(tf.zeros([10]), name='b')

                self.W_assign_value = tf.placeholder(tf.float32, [784, 10], name='W_assign_value')
                self.b_assign_value = tf.placeholder(tf.float32, [10], name='b_assign_value')
                self.W_assign = tf.assign(self.W, self.W_assign_value, name='W_assign')
                self.b_assign = tf.assign(self.b, self.b_assign_value, name='b_assign')

            # output
            with tf.variable_scope('softmax_layer'):
                self.Wx_plus_b = tf.add(tf.matmul(self.x, self.W), self.b, name='Wx_plus_b')
                self.y = tf.nn.softmax(self.Wx_plus_b, name='softmax_OF_Wx_plus_b')

            # labels and training
            with tf.variable_scope('training'):
                self.y_ = tf.placeholder(tf.float32, [None, 10], name='labels')
                self.cross_entropy = tf.reduce_mean( -tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]), name='cross_entropy' )
                self.regularizer_list = [
                    self.alpha * l2(self.W),
                    self.alpha * l2(self.b),
                ]
                self.total_loss = self.cross_entropy + sum(self.regularizer_list)
                self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.total_loss)
            
            # evaluation
            with tf.variable_scope('evaluation'):
                self.correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            
            # summary
            tf.summary.scalar('xentropy', self.cross_entropy, collections=self.summaries)
            tf.summary.scalar('accuracy', self.accuracy, collections=self.summaries)
            self.merge_summaries = tf.summary.merge_all(self.summaries[0])
            
            # get session
            if tf.get_default_session():
                logger.info('default session available; using default session for model')
                self.sess = tf.get_default_session()
            else:
                self.sess = tf.Session()
            
            # initialization
            with tf.variable_scope('initialize'):
                model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_name)
                self.init = tf.variables_initializer(model_variables)
            self.sess.run(self.init)
    
    def get_W(self):
        return self.W.eval(session=self.sess)

    def set_W(self, W_assign_value):
        if not hasattr(self, 'history_W'):
            self.history_W = []
        self.history_W.append( self.get_W() )

        self.sess.run(self.W_assign, feed_dict={self.W_assign_value: W_assign_value})

    def get_b(self):
        return self.b.eval(session=self.sess)

    def set_b(self, b_assign_value):
        if not hasattr(self, 'history_b'):
            self.history_b = []
        self.history_b.append( self.get_b() )

        self.sess.run(self.b_assign, feed_dict={self.b_assign_value: b_assign_value})

    def train_model(self):
        if self.write_summary:
            summary_dir = path.join(path.curdir, 'summary', 'train', self.model_name)
            summary_writer = tf.train.SummaryWriter(summary_dir, self.sess.graph)

        for i in range(self.n_iterations):
            batch_xs, batch_ys = self.mnist_train.next_batch(self.minibatch_size)
            if self.write_summary:
                summary, _ = self.sess.run([self.merge_summaries, self.train_step], feed_dict={self.x: batch_xs, self.y_: batch_ys})
                summary_writer.add_summary(summary, i)
            else:
                self.sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})

    def evaluate_model(self, test_data):
        accuracy_eval = self.sess.run(self.accuracy, feed_dict={self.x: test_data.images, self.y_: test_data.labels})
        return accuracy_eval


class DistSimulation(MNISTSoftmaxRegression):
    
    def __init__(self, n_machines, common_examples_fraction, sync_iterations, averaging_interval, *args, **kwargs):
        self.n_machines = n_machines
        self.common_examples_fraction = common_examples_fraction
        self.sync_iterations = sync_iterations
        self.averaging_interval = averaging_interval

        self._initialize_same = False
        self._sample_with_replacement = False
        self._adaptive_sampling_scheme = False

        super().__init__(*args, **kwargs)

    @property
    def sample_with_replacement(self):
        return self._sample_with_replacement

    @sample_with_replacement.setter
    def sample_with_replacement(self, setting):
        assert isinstance(setting, bool), "setting should be boolean"
        self._sample_with_replacement = setting

    @property
    def adaptive_sampling_scheme(self):
        return self._adaptive_sampling_scheme

    @adaptive_sampling_scheme.setter
    def adaptive_sampling_scheme(self, setting):
        assert isinstance(setting, bool), "setting should be boolean"
        self._adaptive_sampling_scheme = setting

    @property
    def initialize_same(self):
        return self._initialize_same

    @initialize_same.setter
    def initialize_same(self, setting):
        assert isinstance(setting, bool), "setting should be boolean"
        self._initialize_same = setting

    def train_model(self):
        self.partition_data()
        self.train_distributed_models()

    def partition_data(self):
        n_examples = self.mnist_train.images.shape[0]
        random_order = np.random.permutation(n_examples)
        
        n_common_examples = int(self.common_examples_fraction * n_examples)
        n_subset_examples = int( (n_examples-n_common_examples)/self.n_machines )

        common_examples_indices = random_order[0:n_common_examples]
        common_examples = self.mnist_train.images[common_examples_indices, :]
        common_examples_labels = self.mnist_train.labels[common_examples_indices, :]

        if self.sync_iterations:
            n_examples_per_machine = n_common_examples+n_subset_examples
            n_epochs_per_machine = int(np.ceil(self.n_iterations*self.minibatch_size/n_examples_per_machine))
            perm_list = self.get_permutations(n_epochs_per_machine, n_examples_per_machine, n_common_examples, n_subset_examples)
        
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
            data_set = DataSet(images, labels, reshape=False, dtype=dtypes.uint8)
            if self.sync_iterations:
                data_set.perm_list = perm_list
            self.training_data_sets.append( data_set )

    def train_distributed_models(self):
        assert self.averaging_interval <= self.n_iterations, "averaging_interval MUST be <= n_iterations"
        
        self.distributed_models = []
        for i_machine, mnist_train in enumerate(self.training_data_sets):
            dist_model_name = '/'.join([self.model_name, 'dist_model_{0}'.format(i_machine)])

            model = MNISTSoftmaxRegression(self.minibatch_size, self.learning_rate, self.n_iterations, 
                                           mnist_train, model_name=dist_model_name, 
                                           write_summary=self.write_summary,
                                           alpha=self.alpha)
            self.distributed_models.append(model)
        
        # why use the slice [1:] ?
        #    index [0] == 0, zero iterations should not be considered
        train_stride_list = np.arange(0, self.n_iterations+1, self.averaging_interval)[1:]
        for stride_n, train_stride in enumerate(train_stride_list):
            for model in self.distributed_models:
                if stride_n == 0 and not self._initialize_same:
                    logger.debug('initializing distributed models with different values')
                    break
                else:
                    logger.debug('initializing distributed models with same values')
                model.set_W(self.get_W())
                model.set_b(self.get_b())
            for model in self.distributed_models:
                model.train_model()
            self.combine_distributed_models()
    
    def combine_distributed_models(self):
        logger.debug('combine_distributed_models()')

        W_list = []
        b_list = []

        for i_machine, model in enumerate(self.distributed_models):
            W_list.append( model.get_W() )
            b_list.append( model.get_b() )

        W_avg = np.mean(W_list, axis=0)
        b_avg = np.mean(b_list, axis=0)
        self.set_W(W_avg)
        self.set_b(b_avg)
    
    def evaluate_distributed_models(self, test_data):
        accuracy_list = []
        for model in self.distributed_models:
            accuracy_list.append( model.evaluate_model(test_data) )
        return accuracy_list

    def get_permutations(self, num_perms, perm_length, n_common_examples, n_subset_examples):
        perm_list = []
        for perm_n in range(num_perms):
            n_total_examples = n_common_examples + n_subset_examples
            indices = np.arange(n_total_examples)
            if self._adaptive_sampling_scheme:
                # maximum possible probability bias
                #   factor of 0.9 to prevent sampling error for without replacement
                p_bias = n_subset_examples/n_total_examples * 0.9
            else:
                p_bias = 0
            decay_coef = 1
            try:
                p_common_examples = 1/n_total_examples + p_bias/n_common_examples * np.exp(-decay_coef*perm_n)
            except ZeroDivisionError:
                p_common_examples = 0
            try:
                p_subset_examples = 1/n_total_examples - p_bias/n_subset_examples * np.exp(-decay_coef*perm_n)
            except ZeroDivisionError:
                p_subset_examples = 0
            
            assert p_common_examples >= 0 and p_common_examples <= 1, "0 <= p_common_examples <= 1 MUST hold true"
            assert p_subset_examples >= 0 and p_subset_examples <= 1, "0 <= p_subset_examples <= 1 MUST hold true"

            p = np.concatenate([    np.repeat(p_common_examples, n_common_examples),
                                    np.repeat(p_subset_examples, n_subset_examples),
                               ], axis=0)
            
            perm = np.random.choice(indices, perm_length, replace=self._sample_with_replacement, p=p)
        return perm_list

if __name__=='__main__':
    minibatch_size = 1
    learning_rate = 0.01
    n_iterations = 100

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    mnist_classifier = MNISTSoftmaxRegression(minibatch_size, learning_rate, 
                                              n_iterations, mnist.train, 
                                              model_name='UnifiedClassifier', 
                                              write_summary=False,
                                              alpha=0.1)
    mnist_classifier.train_model()
    accuracy = mnist_classifier.evaluate_model(mnist.test)

    logger.info('unified model accuracy = {0}'.format(accuracy))

    n_machines = 4
    common_examples_fraction = 1
    sync_iterations = True
    averaging_interval = n_iterations
    mnist_distributed = DistSimulation(n_machines, common_examples_fraction, sync_iterations,
                                       averaging_interval, 
                                       minibatch_size, learning_rate, n_iterations, 
                                       mnist.train, model_name='DistributedClassifier', 
                                       write_summary=False,
                                       alpha=0.1)
    # mnist_distributed.initialize_same = True
    # mnist_distributed.sample_with_replacement = True
    # mnist_distributed.adaptive_sampling_scheme = True
    mnist_distributed.train_model()
    combined_model_accuracy = mnist_distributed.evaluate_model(mnist.test)
    dist_model_accuracy_list = mnist_distributed.evaluate_distributed_models(mnist.test)

    logger.info('combined_model_accuracy = {0}'.format(combined_model_accuracy))
    logger.info('dist_model_accuracy_list: {0}'.format(dist_model_accuracy_list))
