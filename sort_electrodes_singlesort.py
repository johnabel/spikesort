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
experiment = 'data/032016_1104amstart/'


# we're going to have to combine info from multiple sources. we can't process 
# all of it at once. we can do 10 files at a time tho.


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
    ename : str
        Name of the spike being tested.
    paths : list(str)
        Paths to all the .mcd files being used.
    ----
    """
    # set up inputs
    (ename, dbpath, mcd_idxs, pca_data_noise, gmm_data_noise, noise_std,
             pca_data_sort, gmm_data_sort, sort_std) = inputs
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
            ele1.sort_spikes(sort_data = noise_free_data,
                             sort_times = noise_free_times,
                             pca_param = pca_data_sort,
                             gmm_param = gmm_data_sort,
                             precalc_std = sort_std, method='std', 
                                       )
                                   
            result = [[] for i in range(ele1.num_clusters)] 
            for i in range(ele1.num_clusters):
                result[i] = np.asarray(ele1.neuron_spike_times[str(i)])
        else: 
            result=0    
    except: 
        result = -1
    return result

# still too damn slow. maybe convert to n8umpy before advancing from here
def resample(inputs):
    path, idx, ename, frac = inputs
    d1, t1, s1 = Electrode._load_database_subsample(path, idx, ename, frac)
    return d1, t1, s1

def child_resample(inputs):
    """
    A function which sorts spikes for a single electrode by name. Uses child 
    processes to reduce memory load.
    """
    # use child process to keep it trill
    with futures.ProcessPoolExecutor(max_workers=1) as executor:
        result = executor.submit(resample, inputs).result(timeout=1500) 
        # maybe try timeout errors for parallel?
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
        noise_std = full_ele.calc_standard_deviation
        # sort electrode using EM to categorize ALL spikes
        full_ele.sort_spikes(method='em', precalc_std = noise_std)
        # remove the noise cluster 
        full_ele.remove_noise_cluster()
        noise_free_data = np.vstack(full_ele.neurons.values())
        noise_free_times = np.hstack(full_ele.neuron_spike_times.values())
        
        # sort the noise-free data
        full_ele.fit_gmm(sort_data = noise_free_data,
                             sort_times = noise_free_times, thresh='bics')
        pca_data_sort = full_ele.pca_parameters
        gmm_data_sort = full_ele.gmm_parameters
        sort_std = full_ele.calc_standard_deviation
        neuron_count = full_ele.num_clusters
        full_ele.sort_spikes(sort_data = noise_free_data,
                             sort_times = noise_free_times, 
                             method='std', precalc_std = sort_std)
        # input data for sorting
        chunks = [mcd_labels[x:x+batch_size] for x 
                    in range(0, len(mcd_labels), batch_size)]
        inputs = [
            [ename, dbpath, chunk, pca_data_noise, gmm_data_noise, noise_std,
             pca_data_sort, gmm_data_sort, sort_std]    for chunk in chunks]
        
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
        spike_time_arrays = [np.hstack(firing_times[i]) for i in range(neuron_count)]
                                        
                                        
        # saving the numpy arrays, and the plots
        if savenpy is not False:
            #plots
            if saveplots is True:
                fig2 = full_ele.plot_mean_profile(return_fig=True)
                fig2.savefig(savenpy+ename+'_imgprofiles.png')
                plt.clf()
                plt.close(fig2)
                for pc in range(full_ele.num_comp)[1:]:
                    fig1 = full_ele.plot_clustering(return_fig=True,pc2=pc)
                    fig1.savefig(savenpy+ename+'_imgclusters'+str(pc)+'.png')
                    plt.clf()
                    plt.close(fig1)
                    fig3 = full_ele.plot_heatmap(return_fig=True,  pc2=pc)
                    fig3.savefig(savenpy+ename+'_imgheatmap'+str(pc)+'.png')
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

    # for spike
    enames= np.sort(np.load(experiment+'numpy_database/enames.npy'))
    mcd_labels = np.sort(os.listdir(experiment+'numpy_database'))[:-1]
    # mcd labels is to -1 so as to remove enames.npy from the list.
    
    '''
    # section for resampling, serial ~1.5hr total maybe more
    timer = Electrode.laptimer()
    for ename in enames:
        resample_inputs = [[experiment, label, ename, 0.05] for label 
                                in mcd_labels]
        resampled_data = []
        for res_inp in resample_inputs:
            resampled_data.append(resample(res_inp))
        
        print "Resampled "+ename+"."
        try:
            # setup resampled data
            rda = np.hstack([d[0].T for d in resampled_data]).T
            tda = np.hstack(d[1] for d in resampled_data)
            np.save(experiment+'resampled_test_sets/'+ename+'_rda.npy', rda)
            np.save(experiment+'resampled_test_sets/'+ename+'_tda.npy', tda)
        except: 
            print "No spikes identified for "+ename +"."
            #expected for 15A, 17B, 26B, 27B, 28b, 38B, 46B, 47B, 48B     
    print "Total resampling time: "+str(np.round(timer(), 2))+"."
    '''
    
    # section for sorting all spikes.
    timer = Electrode.laptimer()
    neuron_count = []
    enames_b = []
    for ename in enames:    
        if ename[-1]=='A':
            enames_b.append(ename)
            try:
                # LOAD RESAMPLED DATA
                rda = np.load(experiment+'resampled_test_sets/'+ename+'_rda.npy')
                tda = np.load(experiment+'resampled_test_sets/'+ename+'_tda.npy')
                full_ele = Electrode.Electrode(ename)# ignore all the names etc
                full_ele.load_array_data(rda, tda)
                
                # SORT THE SPIKES
                spks_sorted, nc = sort_electrode(
                   ename, experiment, mcd_labels, 10, full_ele, 
                   savenpy=experiment+'neurons_singlesort_A/',
                   saveplots=True)
                   
                print "Sorted for "+str(ename)
            except IOError:
                print "No spikes found for channel "+ename[-3:]
        else: 
            print "Sorting ignored for "+ename # currently ignoring A
    enamcount = map(list, zip(*[enames_b, neuron_count]))
    print ("Time to serially sort all spikes "
                    +str(round(timer(),2))+"s.")
    
    # define plots
    def plot_frate(spike_times, window=600):
        times,frates = Electrode.firing_rates(spike_times, win=window)
        fig = plt.figure()
        plt.plot(times/3600, frates,'k.')
        plt.xlim([0,np.max(times/3600)])
        plt.xlabel('Time (h)')
        plt.ylabel('Freq. (Hz)')
        return fig
    
    def plot_isi(spike_times):
        isi = np.diff(spike_times)
        too_hi_isi = 100*np.round(len(np.where((1/isi)>200))/ len(isi),4)
        fig = plt.figure()
        plt.plot(spike_times[:-1]/3600, 1/isi,'k.', label='1/ISI, '+str(too_hi_isi)+'% > 200Hz')
        plt.xlabel('Time (h)')
        plt.xlim([0,np.max(spike_times/3600)])
        plt.ylabel('1/ISI')
        plt.ylim([0,50])
        plt.legend()
        return fig
        
    
    # save all files to csv
    result_path = experiment+'neurons_singlesort_A/'
    nfiles = np.sort(os.listdir(result_path))
    for nn in nfiles:
        if nn[-3:]=='npy' and nn[4]=='n':
            times = np.load(result_path+nn)
            if len(times) > 5E3:
                try:
                    np.savetxt(result_path+'csv/'+nn[:-4]+'.csv', times, delimiter=',')
                    #save plots
                    fig1 = plot_frate(times)
                    fig1.savefig(result_path+'csv/'+nn[:11]+'_rate.png')
                    plt.clf()
                    plt.close(fig1)
                    fig2 = plot_isi(times)
                    fig2.savefig(result_path+'csv/'+nn[:11]+'_isi.png')
                    plt.clf()
                    plt.close(fig2)
                except: print 'failed for'+nn
            
            
            
