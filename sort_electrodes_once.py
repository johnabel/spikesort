# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:48:41 2016

@author: abel, probably mazuski
"""

from __future__ import division

import itertools
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
experiment = 'data/rishabh_selected/'
enames= np.sort(np.load(experiment+'numpy_database/enames.npy'))
stim = None




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
    (ename, dbpath, mcd_idxs, pca_data_noise, gmm_data_noise, noise_std) = inputs
    try:
        #remove noise cluster
        ele1 = Electrode.Electrode(ename, database_path=dbpath, 
                                   mcd_idxs=mcd_idxs) 
                                   
        #categorize the spikes using em
        ele1.sort_spikes(pca_param=pca_data_noise, gmm_param=gmm_data_noise, 
                         method='em', precalc_std= noise_std)
        # sort the results
        if ele1.num_clusters > 1:
            noise_free_data = np.vstack(ele1.neurons.values())
            noise_free_times = np.hstack(ele1.neuron_spike_times.values())
            ele1.sort_spikes(sort_data=noise_free_data,
                             sort_times=noise_free_times,
                             pca_param=pca_data_noise, 
                             gmm_param=gmm_data_noise,
                             precalc_std= noise_std, 
                             method='em'#'std', 
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

def sort_electrode(ename, dbpath, mcd_labels, batch_size, full_ele, bics_thresh=5000,
                   savenpy=False, saveplots=False, recombine_neighbors=False):
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
    full_ele.fit_gmm(thresh='bics', bics_thresh=bics_thresh)
    cluster_count = full_ele.num_clusters
    if cluster_count == 1:
        # there is only noise
        return [], [], 0
    
    else:
        #deconstruct pca, gmm
        pca_data_noise = full_ele.pca_parameters
        gmm_data_noise = full_ele.gmm_parameters
        precalc_std = full_ele.calc_standard_deviation
        # sort electrode using expectation maximizatn to categorize ALL spikes
        full_ele.sort_spikes(method='em', precalc_std = precalc_std)
        
        
        # use the discovered sorting
        # input data for sorting
        chunks = [mcd_labels[x:x+batch_size] for x 
                    in range(0, len(mcd_labels), batch_size)]
        inputs = [
            [ename, dbpath, chunk, pca_data_noise, gmm_data_noise, precalc_std
             ]    for chunk in chunks]
        
        # feed the inputs to the categorization function
        outputs = []
        for inputi in inputs[:]:
            # consider parallelizing
            outputs.append(child_categorize(inputi))
        
        # get the outputs into a reasonable format
        firing_times = [[] for _ in range(cluster_count)]
        spike_shapes = [[] for _ in range(cluster_count)]
        for idx,output in enumerate(outputs):
            if output==-1: pass # ignore if no output
            else:
                for cidx in range(cluster_count):
                    firing_times[cidx].append(output[0][cidx])
                    spike_shapes[cidx].append(output[1][cidx])
        
        
        # option to combine spikes that have a very similar shape
        if recombine_neighbors:
            # set property false
            converged = False

            while converged is False:
                mean_shapes = [np.mean(np.vstack(shape),0) for 
                                            shape in spike_shapes]
                neurons_first = np.arange(cluster_count)
                neurons_second = np.arange(cluster_count)
                # finds which are most similar
                sim_mat = np.zeros((len(neurons_first), len(neurons_second)))
                for comparison in itertools.product(neurons_first, neurons_second):
                    c0 = comparison[0]; c1 = comparison[1]
                    shape1 = mean_shapes[c0]
                    shape2 = mean_shapes[c1]
                    #set up the similarity matrix
                    sim_mat[np.where(neurons_first==c0)[0][0], 
                            np.where(neurons_second==c1)[0][0]]=\
                                            np.sum(np.abs(shape1-shape2))
                # ignore diags
                sim_mat = sim_mat+np.diag([1]*cluster_count)
                min_loc = np.unravel_index(sim_mat.argmin(), sim_mat.shape)
                
                if sim_mat[min_loc]>1E-4: # threshold for spike similarity
                    converged=True
                else:
                    # combine the most similar one, start off the new lists 
                    # with it
                    spike_shapes_new = []
                    firing_times_new = []
                    combined_times = np.hstack([firing_times[min_loc[0]],
                                                firing_times[min_loc[1]]]).flatten()
                    combined_spikes = np.vstack([spike_shapes[min_loc[0]][0],
                                                spike_shapes[min_loc[1]]][0])
                    spike_shapes_new.append(combined_spikes)
                    firing_times_new.append(combined_times)
                    
                    # add again the ones that were not combined
                    non_combined = list(set(neurons_first)-set(min_loc))
                    for nc_cluster in non_combined:
                        spike_shapes_new.append(spike_shapes[nc_cluster])
                        firing_times_new.append(firing_times[nc_cluster])
                        
                    cluster_count = len(spike_shapes_new)
                    spike_shapes = spike_shapes_new
                    firing_times = firing_times_new
                    
                    

        # once again, if only noise exists
        cluster_count = len(spike_shapes)
        if cluster_count == 1:
            # there is only noise
            return [], [], 0
        else:
            # one cluster is noise, identify which
            neuron_count = cluster_count -1
            mean_shapes = [np.mean(np.vstack(shape),0) for 
                                            shape in spike_shapes]
            profile_absmean = []
            for i in range(cluster_count):
                profile_absmean.append(np.abs(mean_shapes[i]).mean())
            remove_cluster = np.argmin(profile_absmean)
            
            
            # get into finalized format, remove the noise
            spike_time_arrays = []
            spike_shape_arrays = []
            for i in range(cluster_count):
                # remove noise
                if i != remove_cluster:
                    if len(firing_times[i])>0:
                        spike_time_arrays.append(np.hstack(firing_times[i]))
                        spike_shape_arrays.append(np.vstack(spike_shapes[i]))
                    else:
                        spike_time_arrays.append([])
            
            
            # plotting_electrode is the electrode object used for plots
            plot_ele = Electrode.Electrode(ename)
            plot_ele.gmm = 'decoy' # just to fulfill a requirement of the obj
            plot_ele.neurons = {}
            for nidx in range(neuron_count):
                plot_ele.neurons[nidx] = spike_shape_arrays[nidx]
            
            
            # saving the numpy arrays, and the plots
            if savenpy is not False:
                #plots
                if saveplots is not False:
                    fig2, waveforms = full_ele.plot_mean_profile(
                                                             return_fig=True)
                    fig2.savefig(saveplots+ename+'_all_spike_profiles.png')
                    plt.clf()
                    plt.close(fig2)
                    fig4, waveforms = plot_ele.plot_mean_profile(
                                                             return_fig=True)
                    fig4.savefig(saveplots+ename+'_combined_noiseless_profiles.png')
                    plt.clf()
                    plt.close(fig2)
                    for pc in range(full_ele.num_comp)[1:]:
                        fig1 = full_ele.plot_clustering(return_fig=True,pc2=pc)
                        fig1.savefig(saveplots+ename+'_'+str(pc)+
                                                'pc_clusters.png')
                        plt.clf()
                        plt.close(fig1)
                        fig3 = full_ele.plot_heatmap(return_fig=True,  pc2=pc)
                        fig3.savefig(saveplots+ename+'_'+str(pc)+
                                                'pc_heatmap.png')
                        plt.clf()
                        plt.close(fig3)
    
                # numpy arrays
                for i in range(neuron_count):
                    np.save(savenpy+ename+'_n'+str(i)+'_profiles.npy',
                                spike_shape_arrays[i])
                    np.save(savenpy+ename+'_n'+str(i)+'_times.npy',
                                spike_time_arrays[i])
                    np.savetxt(saveplots+ename+
                                    '_waveform.csv', waveforms, delimiter=',')
                                
            return spike_time_arrays, spike_shape_arrays, neuron_count
    



if __name__=='__main__':
    # Set up the directories for saving the results
    # if the folder exists, this prevents it from being overwritten
    # if you want to overwrite it, just delete it.
    
    if os.path.isdir(experiment+'numpy_neurons'):
        print ("Numpy neurons already exists in "+experiment+
                ". Please delete or select a new location.")
        import sys
        sys.exit()
    else: os.mkdir(experiment+'numpy_neurons')
    
        
    subdirectories = np.sort(os.listdir(experiment+'numpy_database'))[:-1]
    
    for diridx, directory in enumerate(subdirectories):
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
        
    
        bics_selected = 5000
        
        # section for sorting all spikes.
        timer = Electrode.laptimer()
        neuron_count = []
        subsample_path = experiment+'subsampled_test_sets/'+directory+'/'
        for ename in enames:    
            try:
                # LOAD RESAMPLED DATA
                rda = np.load(subsample_path+ename+'_rda.npy')
                tda = np.load(subsample_path+ename+'_tda.npy')
                full_ele = Electrode.Electrode(ename)# ignore all the names etc
                full_ele.load_array_data(rda, tda)
                
                # SORT THE SPIKES
                spks_sorted, spks_shapes, nc = sort_electrode(
                   ename, experiment+'numpy_database/'+directory+'/',
                   mcd_labels, 10, full_ele, bics_thresh = bics_selected,
                   savenpy=experiment+'numpy_neurons/'+directory+'/',
                   saveplots=experiment+'numpy_neurons/'+directory+'/sorting_plots/',
                   recombine_neighbors=True)
                   
                print "Sorted for "+str(ename)
            except IOError:
                print "No spikes found for channel "+ename[-3:]
    
        print ("Time to serially sort all spikes "
                        +str(round(timer(),2))+"s.")
        

            

            
            
