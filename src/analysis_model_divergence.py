#!/usr/bin/python3
import scipy.io as sio
import numpy as np
import pickle
import logging
import pdb
from concurrent.futures import ProcessPoolExecutor

import input_data
import sys
import ilogger
import time

ilogger.setup_root_logger('/dev/null', logging.ERROR)

class Metrics(object):
    def __init__(self, name):
        self.name = name

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


def l2_error(weight1, weight2):
    return np.linalg.norm(weight1-weight2)

def calculate_mean_error(weight_list_1, weight_list_2):
    mean_error = 0
    for w1, w2 in zip(weight_list_1, weight_list_2):
        mean_error += l2_error(w1, w2)
    mean_error /= len(weight_list_1)
    return mean_error

def get_dist_model_distance(dist_model):
    n_dist_models = len(dist_model.distributed_models)
    model_dist_matrix = np.zeros((n_dist_models, n_dist_models))

    for i in range(0, n_dist_models):
        for j in range(i+1, n_dist_models):
            model1 = dist_model.distributed_models[i]
            model2 = dist_model.distributed_models[j]

            weight_list_1 = model1.get_W_list_values()
            weight_list_2 = model2.get_W_list_values()
            weight_list_1.extend(model1.get_b_list_values())
            weight_list_2.extend(model2.get_b_list_values())

            model_distance = calculate_mean_error(weight_list_1, weight_list_2)
            model_dist_matrix[i,j] = model_distance
            model_dist_matrix[j,i] = model_distance
    return model_dist_matrix

def run_simulation(features, mnist, model_args, model_kwargs):
    import mnist_distributed_sim_convex as Convex
    import mnist_distributed_sim_nonconvex as NonConvex
    import mnist_distributed_sim_strongly_convex as StronglyConvex

    model_type_dict = {
        'Convex': Convex,
        'NonConvex': NonConvex,
        'StronglyConvex': StronglyConvex,
    }
    DistSimulation = model_type_dict[features[0]].DistSimulation

    mnist_distributed = DistSimulation(*model_args, **model_kwargs)
    mnist_distributed.initialize_same = init_same
    mnist_distributed.sample_with_replacement = sample_with_replacement
    mnist_distributed.adaptive_sampling_scheme = adap_sampling
    mnist_distributed.train_model()
    
    dist_model_distance_matrix = get_dist_model_distance(mnist_distributed)
    dist_model_accuracy_list = mnist_distributed.evaluate_distributed_models(mnist.test)
    combined_model_accuracy = mnist_distributed.evaluate_model(mnist.test)

    # print(dist_model_distance_matrix)
    # print(dist_model_accuracy_list)
    # print(combined_model_accuracy)

    return (features, dist_model_distance_matrix, dist_model_accuracy_list, combined_model_accuracy)
    

if __name__=='__main__':
    # model_type_list = ['Convex', 'NonConvex', 'StronglyConvex']
    model_type_list = ['StronglyConvex']
    n_iterations_list = [100, 500, 1000, 5000, 10000, 55000]
    common_ratio_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    init_same_list = [False, True]
    adap_sampling_list = [False, True]
    repetitions = 5
    # model_type_list = ['Convex'] # , 'NonConvex', 'StronglyConvex']
    # n_iterations_list = [10000]
    # common_ratio_list = [0.0, 1.0]
    # init_same_list = [False]
    # adap_sampling_list = [False, True]
    # repetitions = 2

    n_machines = 4
    learning_rate = 0.01
    minibatch_size = 1
    sync_iterations = True
    sample_with_replacement = False
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    model_distance_dict = {}
    dist_accuracy_dict = {}
    combined_accuracy_dict = {}
    futures_list = []

    try:
        for model_type in model_type_list:
            with ProcessPoolExecutor(max_workers=10) as executor:
                print(model_type)
                for n_iterations in n_iterations_list:
                    averaging_interval = n_iterations
                    for common_ratio in common_ratio_list:
                        for init_same in init_same_list:
                            for adap_sampling in adap_sampling_list:
                                features = (model_type, n_iterations, common_ratio, init_same, adap_sampling)
                                model_distance_dict[features] = []
                                dist_accuracy_dict[features] = []
                                combined_accuracy_dict[features] = []
                                
                                model_args = [n_machines, common_ratio, sync_iterations, averaging_interval, 
                                              minibatch_size, learning_rate, n_iterations, mnist.train]
                                model_kwargs = {
                                    'model_name': 'DistributedClassifier', 
                                    'write_summary': False,
                                }
                                for rep in range(repetitions):
                                    future = executor.submit(run_simulation, features, mnist, model_args, model_kwargs)
                                    futures_list.append(future)
    except:
        pass

    # control will only come here after all executors are finished executing
    for future in futures_list:
        try:
            features, dist_model_distance_matrix, dist_model_accuracy_list, combined_model_accuracy = future.result()
        except:
            continue
        model_distance_dict[features].append(dist_model_distance_matrix)
        dist_accuracy_dict[features].append(dist_model_accuracy_list)
        combined_accuracy_dict[features].append(combined_model_accuracy)
    
    data_attribute_list = ['model_type', 'n_iterations', 'common_ratio', 'init_same', 'adap_sampling']

    model_distance_metrics = Metrics('model_distance')
    model_distance_metrics.data = model_distance_dict
    model_distance_metrics.data_attribute_list = data_attribute_list

    dist_accuracy_metrics = Metrics('dist_accuracy')
    dist_accuracy_metrics.data = dist_accuracy_dict
    dist_accuracy_metrics.data_attribute_list = data_attribute_list

    combined_accuracy_metrics = Metrics('combined_accuracy')
    combined_accuracy_metrics.data = combined_accuracy_dict
    combined_accuracy_metrics.data_attribute_list = data_attribute_list

    for metric in [model_distance_metrics, dist_accuracy_metrics, combined_accuracy_metrics]:
        metric.n_machines = n_machines
        metric.learning_rate = learning_rate
        metric.minibatch_size = minibatch_size
        metric.sync_iterations = sync_iterations
        metric.sample_with_replacement = sample_with_replacement

    with open('/mydata/model_distance_3.pickle', 'wb') as f:
        pickle.dump(model_distance_metrics, f)
    with open('/mydata/dist_accuracy_3.pickle', 'wb') as f:
        pickle.dump(dist_accuracy_metrics, f)
    with open('/mydata/combined_accuracy_3.pickle', 'wb') as f:
        pickle.dump(combined_accuracy_metrics, f)
