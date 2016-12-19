#!/usr/bin/python3
import scipy.io as sio
import numpy as np
import pickle
import logging
import pdb
from concurrent.futures import ProcessPoolExecutor

from analysis_model_divergence import Metrics

import input_data
import sys
import ilogger
import time

ilogger.setup_root_logger('/dev/null', logging.ERROR)

def run_simulation(features):
    import mnist_distributed_sim_convex as Convex
    import mnist_distributed_sim_nonconvex as NonConvex
    import mnist_distributed_sim_strongly_convex as StronglyConvex
    
    unified_model_type_dict = {
        'Convex': Convex.MNISTSoftmaxRegression,
        'NonConvex': NonConvex.MNISTConvNet,
        'StronglyConvex': StronglyConvex.MNISTSoftmaxRegression,
    }
    dist_model_type_dict = {
        'Convex': Convex.DistSimulation,
        'NonConvex': NonConvex.DistSimulation,
        'StronglyConvex': StronglyConvex.DistSimulation,
    }
    UnifiedModel = unified_model_type_dict[model_type]
    DistSimulation = dist_model_type_dict[model_type]

    print('Unified Model')
    print(unified_model_args)
    print(unified_model_kwargs)
    print('Dist Simulation')
    print(dist_model_args)
    print(dist_model_kwargs)

    mnist_classifier = UnifiedModel(*unified_model_args, **unified_model_kwargs)
    mnist_classifier.test_data = mnist.test
    mnist_classifier.train_model()

    mnist_distributed = DistSimulation(*dist_model_args, **dist_model_kwargs)
    mnist_distributed.test_data = mnist.test
    mnist_distributed.initialize_same = init_same
    # mnist_distributed.sample_with_replacement = True
    mnist_distributed.adaptive_sampling_scheme = adap_sampling
    mnist_distributed.train_model()

    metric = Metrics('model_averaging')
    metric.model_type = model_type
    metric.minibatch_size = minibatch_size
    metric.learning_rate = learning_rate
    metric.adaptive_learning_rate = False
    metric.n_iterations = n_iterations
    metric.n_machines = n_machines
    metric.common_examples_fraction = common_examples_fraction
    metric.sync_iterations = sync_iterations
    metric.averaging_interval = averaging_interval
    
    metric.initialize_same = mnist_distributed.initialize_same
    metric.adaptive_sampling_scheme = mnist_distributed.adaptive_sampling_scheme
    
    metric.unified_history_accuracy = mnist_classifier.history_accuracy
    metric.history_accuracy = mnist_distributed.history_accuracy
    metric.dist_model_distance = mnist_distributed.history_dist_model_distance
    metric.dist_history_accuracy = [ model.history_accuracy for model in mnist_distributed.distributed_models ]
    
    metric.unified_W = mnist_classifier.get_W_list_values()
    metric.unified_b = mnist_classifier.get_b_list_values()
    metric.history_W = mnist_distributed.history_W + [ mnist_distributed.get_W_list_values() ]
    metric.history_b = mnist_distributed.history_b + [ mnist_distributed.get_b_list_values() ]
    metric.dist_history_W = [ model.history_W + [ model.get_W_list_values() ] for model in mnist_distributed.distributed_models ]
    metric.dist_history_b = [ model.history_b + [ model.get_b_list_values() ] for model in mnist_distributed.distributed_models ]

    return metric

if __name__=='__main__':
    model_type_list = ['NonConvex']
    n_iterations_list = [10000]
    common_ratio_list = [0.0, 0.4, 1.0]
    init_same_list = [True]
    adap_sampling_list = [False, True]
    
    n_machines = 4
    learning_rate = 0.01
    minibatch_size = 1
    sync_iterations = True
    sample_with_replacement = False
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    futures_list = []

    try:
        for model_type in model_type_list:
            with ProcessPoolExecutor(max_workers=10) as executor:
                print(model_type)
                for n_iterations in n_iterations_list:
                    for averaging_interval in [n_iterations//8, n_iterations//4, n_iterations//2]:
                        for common_ratio in common_ratio_list:
                            for init_same in init_same_list:
                                for adap_sampling in adap_sampling_list:
                                    
                                    unified_model_args = [
                                        minibatch_size, learning_rate, n_iterations, mnist.train,
                                    ]
                                    unified_model_kwargs = {
                                        'model_name': 'UnifiedClassifier',
                                        'write_summary': False,
                                    }
                                    dist_model_args = [
                                        n_machines, common_ratio, sync_iterations, averaging_interval, 
                                        minibatch_size, learning_rate, n_iterations, mnist.train,
                                    ]
                                    dist_model_kwargs = {
                                        'model_name': 'DistributedClassifier', 
                                        'write_summary': False,
                                    }
                                    future = executor.submit(run_simulation)
                                    futures_list.append(future)
    except:
        pass

    # control will only come here after all executors are finished executing
    index_offset = 2
    for i, future in enumerate(futures_list):
        try:
            metric = future.result()
        except:
            continue
        fname = '{0}_model_averaging_{1}.pickle'.format(metric.model_type, index_offset+i)
        with open(fname, 'wb') as f:
            pickle.dump(metric, f)
