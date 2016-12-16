from __future__ import division

import numpy  as np
import scipy as sp
from sklearn import decomposition, cluster
import neuroshare as ns
import Electrode
from time import time
from concurrent import futures # requires pip install futures
import os
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt


#trial data
experiment = 'data/stimulation_included/'
enames= np.sort(np.load(experiment+'numpy_database/enames.npy'))[:10]




def child_categorize(inputs):
    """
    A function which sorts spikes for a single electrode by name. Uses child 
    processes to reduce memory load.
    """
    # use child process to keep it trill
    with futures.ProcessPoolExecutor(max_workers=1) as executor:
        result = executor.submit(categorize, inputs).result()
    return result

def categorize(inputs):
    """
    A function which sorts spikes for a single electrode by name.
    
    ----
        (ename, dbpath, mcd_idxs, pca_data_noise, gmm_data_noise, noise_std,
             pca_tree, gmm_tree, std_tree) = inputs
    ----
    """
    # set up inputs
    (ename, dbpath, mcd_idxs, pca_data_noise, gmm_data_noise, noise_std,
             pca_tree, gmm_tree, std_tree) = inputs
    try:
        #remove noise cluster
        ele1 = Electrode.Electrode(ename, database_path=dbpath, 
                                   mcd_idxs=mcd_idxs) 
                                   
        #categorize the spikes using em
        ele1.sort_spikes(pca_param=pca_data_noise, gmm_param=gmm_data_noise, 
                         method='em', precalc_std= noise_std)
        ele1.remove_noise_cluster(method='mean_profile')
        # sort the results
        if ele1.num_clusters > 1:
            noise_free_data = np.vstack(ele1.neurons.values())
            noise_free_times = np.hstack(ele1.neuron_spike_times.values())
            ele1.recursive_sort_spikes(noise_free_data,
                                       noise_free_times,
                                       pca_tree,
                                       gmm_tree,
                                       std_tree, final_method='std', 
                                       )
                                   
            result = [[[] for i in range(ele1.num_clusters)], 
                       [[] for i in range(ele1.num_clusters)]] 
            for i in range(ele1.num_clusters):
                result[0][i] = np.asarray(ele1.neuron_spike_times[str(i)])
                result[1][i] = np.asarray(ele1.neurons[str(i)])
        else: 
            result=0    
    except: 
        result = -1
    return result


if os.path.isdir(experiment+'numpy_neurons'):
    print ("Numpy neurons already exists in "+experiment+
            ". Please delete or select a new location.")
    import sys
    sys.exit()
else: os.mkdir(experiment+'numpy_neurons')

    
subdirectories = np.sort(os.listdir(experiment+'numpy_database'))[:-1]

directory = subdirectories[0]
files_in_folder = np.sort(os.listdir(experiment+'numpy_database/'
                            +directory))
mcd_labels = []
for filei in files_in_folder:
    if filei[0]=='.':
        pass
    else:
        mcd_labels.append(filei)

# save all files to csv, save plots
result_path = experiment+'numpy_neurons/'+directory+'/'
os.mkdir(result_path)
if os.path.isdir(result_path+'/sorting_plots'):
    print ("Sorting results already exists in "+experiment+
            ". Please delete or select a new location.")
    import sys
    sys.exit()
else: 
    os.mkdir(result_path+'/sorting_plots') 


# section for sorting all spikes.
timer = Electrode.laptimer()
neuron_count = []
subsample_path = experiment+'subsampled_test_sets/'+directory+'/'

ename = enames[0]
rda = np.load(subsample_path+ename+'_rda.npy')
tda = np.load(subsample_path+ename+'_tda.npy')
full_ele = Electrode.Electrode(ename)# ignore all the names etc
full_ele.load_array_data(rda, tda)

full_ele.fit_gmm(thresh='bics')

dbpath = experiment+'numpy_database/'+directory+'/'
batch_size = 3

cluster_count = full_ele.num_clusters
if cluster_count == 1:
    # there is only noise
    pass
    #return [], [], 0

else:
    #deconstruct pca, gmm
    pca_data_noise = full_ele.pca_parameters
    gmm_data_noise = full_ele.gmm_parameters
    precalc_std = full_ele.calc_standard_deviation
    # sort electrode using expectation maximizatn to categorize ALL spikes
    full_ele.sort_spikes(method='em', precalc_std = precalc_std)
    # remove the noise cluster 
    full_ele.remove_noise_cluster()
    noise_free_data = np.vstack(full_ele.neurons.values())
    noise_free_times = np.hstack(full_ele.neuron_spike_times.values())
    
    # do a recursive sorting f the remaining spikes
    pca_tree, gmm_tree, std_tree = full_ele.recursive_fit_gmm(
                        noise_free_data, noise_free_times, pca_data_noise)
    full_ele.recursive_sort_spikes(noise_free_data, noise_free_times, 
                          pca_tree, gmm_tree, std_tree, final_method='std')
    neuron_count = full_ele.num_clusters
    
    # use the discovered sorting (trees) to now sort ALL the data
    # input data for sorting
    chunks = [mcd_labels[x:x+batch_size] for x 
                in range(0, len(mcd_labels), batch_size)]
    inputs = [
        [ename, dbpath, chunk, pca_data_noise, gmm_data_noise, precalc_std,
         pca_tree, gmm_tree, std_tree]    for chunk in chunks]
    
    # feed the inputs to the categorization function
    outputs = []
    for inputi in inputs[:]:
        # consider parallelizing
        outputs.append(child_categorize(inputi))
    
    # get the outputs into a reasonable format
    firing_times = [[] for _ in range(neuron_count)]
    spike_shapes = [[] for _ in range(neuron_count)]
    for idx,output in enumerate(outputs):
        if output==-1: pass # ignore if no output
        else:
            for neuron_index in range(neuron_count):
                firing_times[neuron_index].append(
                        output[0][neuron_index])
                spike_shapes[neuron_index].append(output[1][neuron_index])
    spike_time_arrays = []
    spike_shape_arrays = []






    