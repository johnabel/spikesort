# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:48:41 2016

@author: abel, probably mazuski
"""

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
experiment = 'data/mega_clusters/'



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
                                   
            result = [[] for i in range(ele1.num_clusters)] 
            for i in range(ele1.num_clusters):
                result[i] = np.asarray(ele1.neuron_spike_times[str(i)])
        else: 
            result=0    
    except: 
        result = -1
    return result

def sort_electrode(ename, dbpath, mcd_labels, batch_size, full_ele, 
                   savenpy=False, saveplots=False):
    """
    Function for sorting a single electrode completely. Currently only serial.
    
    ----
    ename : str
        Name of the spike being tested.
    paths : list(str)
        Paths to all the .mcd files being used.
    batch_size : int
        Number of files to process at once. I usually use 10, but this does
        create a large RAM load
    full_ele : Electrode.Electrode
        The electrode which has been resampled across all times. This function
        handles the spike sorting, and recursive sorting.
    savenpy : str (optional, defaults to false)
        Will save a numpy array of each neuron's firing times to this directory
    ----
    """
    # fit a gmm to remove the noise
    full_ele.fit_gmm(thresh='bics')
    cluster_count = full_ele.num_clusters
    if cluster_count == 1:
        # there is only noise
        return [], 0
    
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
        
        if neuron_count > 7: #tunable
            pca_tree, gmm_tree, std_tree = full_ele.recursive_fit_gmm(
                                noise_free_data, noise_free_times, 
                                pca_data_noise, bics_thresh=10000)
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
        for idx,output in enumerate(outputs):
            if output==-1: pass # ignore if no output
            else:
                for neuron_index in range(neuron_count):
                    firing_times[neuron_index].append(
                            output[neuron_index])
        spike_time_arrays = []
        for i in range(neuron_count):
            if len(firing_times[i])>0:
                spike_time_arrays.append(np.hstack(firing_times[i]))
            else:
                spike_times_arrays.append(np.array([0]))
                                        
                                        
        # saving the numpy arrays, and the plots
        if savenpy is not False:
            #plots
            if saveplots is not False:
                fig2, waveforms = full_ele.plot_mean_profile(return_fig=True)
                fig2.savefig(saveplots+ename+'_spike_profiles.png')
                plt.clf()
                plt.close(fig2)
                fig3 = full_ele.plot_through_time(noise_free_times, return_fig=True)
                fig3.savefig(saveplots+ename+'_spikes_through_time.png')
                np.savetxt(saveplots+ename+
                                '_waveform.csv', waveforms, delimiter=',')
                plt.clf()
                plt.close(fig3)
                for pc in range(full_ele.num_comp)[1:]:
                    fig1 = full_ele.plot_clustering(return_fig=True,pc2=pc)
                    fig1.savefig(saveplots+ename+str(pc)+'pc_clusters.png')
                    plt.clf()
                    plt.close(fig1)
                    fig3 = full_ele.plot_heatmap(return_fig=True,  pc2=pc)
                    fig3.savefig(saveplots+ename+str(pc)+'pc_heatmap.png')
                    plt.clf()
                    plt.close(fig3)

            # numpy arrays
            for i in range(neuron_count):
                np.save(savenpy+ename+'_cluster'+str(i)+'_profile.npy',
                            full_ele.neurons[str(i)].mean(0))
                np.save(savenpy+ename+'_neuron'+str(i)+'_times.npy',
                            spike_time_arrays[i])
                            
        return spike_time_arrays, neuron_count
    



if __name__=='__main__':
    # Set up the directories for saving the results
    # if the folder exists, this prevents it from being overwritten
    # if you want to overwrite it, just delete it.
    
    experiment = 'data/mega_clusters/'
    enames = ['33','35','75']
    
    
    # section for sorting all spikes.
    timer = Electrode.laptimer()
    neuron_count = []
    for ename in enames:    
        try:
            # LOAD RESAMPLED DATA
            rda = np.load(experiment+'subsampled_test_sets/'+ename+'_rda.npy')
            tda = np.load(experiment+'subsampled_test_sets/'+ename+'_tda.npy')
            full_ele = Electrode.Electrode(ename)# ignore all the names etc
            full_ele.load_array_data(rda, tda)
            
            # SORT THE SPIKES
            full_ele.fit_gmm(thresh='bics')
            cluster_count = full_ele.num_clusters
            if cluster_count == 1:
                # there is only noise
                print 0
            
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
                print "prelim count ="+str(neuron_count)
                
                iterate = 0
                while neuron_count > 7: #tunable
                    pca_tree, gmm_tree, std_tree = full_ele.recursive_fit_gmm(
                                        noise_free_data, noise_free_times, 
                                        pca_data_noise, bics_thresh=5000000)
                    full_ele.recursive_sort_spikes(noise_free_data, noise_free_times, 
                                          pca_tree, gmm_tree, std_tree, final_method='std')
                    neuron_count = full_ele.num_clusters
                    print "revised count ="+str(neuron_count)
                    iterate+=1

    print ("Time to serially sort all spikes "
                    +str(round(timer(),2))+"s.")
    

            
            
