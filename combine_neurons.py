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
import itertools



import matplotlib as mpl
import matplotlib.pyplot as plt


#trial data
experiment = 'data/example/'
database_path = experiment # put subsamples in this spot
result_path = experiment+"numpy_neurons"
subdirs = np.sort(os.listdir(experiment+'numpy_database'))[:-1]
stim = None # none if no stimulus file, otherwise set to name of stimulus

# define plots
def plot_frate(spike_times, window=600):
    times,frates = Electrode.firing_rates(spike_times, win=window)
    fig = plt.figure()
    plt.plot(times/3600, frates,'k.')
    plt.xlim([0,np.max(times/3600)])
    plt.xlabel('Time (h)')
    plt.tight_layout()
    plt.ylabel('10-min Mean Freq. (Hz)')
    return fig

def plot_isi(spike_times):
    isi = np.diff(spike_times)
    isi_millis = 1/isi
    fig = plt.figure()
    ax = plt.subplot()
    ax.plot(spike_times[:-1]/3600,isi_millis,'k.', alpha=0.1)
    ax.set_ylabel('1/ISI (Hz)')
    ax.set_ylim([0,20])
    ax.set_xlabel('Time (h)')
    plt.tight_layout()
    return fig

def plot_isi_min(spike_times):
    isi = np.diff(spike_times)
    fig = plt.figure()
    ax = plt.subplot()
    ax.plot(spike_times[:-1]/3600,isi,'.', alpha=0.1)
    ax.set_xlabel('Time (h)')
    #ax.set_xscale('log')
    ax.set_ylabel('ISI (s)')
    ax.set_ylim([0,1])
    plt.tight_layout()
    return fig

def combine_neurons_stim(ename, subdirs, stim):
    """ takes in electrode name, subdirectories, and the subdirectory that 
    contains the stimulus.
    
    The subdirectories should be approx. 1 day long, and every neuron should be
    present in every subdirectory. This function stitches full neurons together
    by matching across each subdirectionry EXCEPT the stimulus, and creating 
    full time neuron trajectories.
    
    The stimulus is then compared to the subdirectory immediately preceding it
    in order to tell which neurons were stimulated.
    
    The combined neurons folder then will have folders for each electrode. In 
    the folders, there will be 2-4 files for each neuron:
    - spike times for the neuron (ignoring stimulus)
    - stimulus spike times for the neuron (if exists)
    - spike shapes for the neuron ignoring stimulus
    - spike shapes during the stimulus (if exists)
    Also, a csv will be included for showing how each neuron matches up across
    time.
    
    """
    
    # figure out where the stimulus is, set subdirs to ignore stimulus and 
    # stim_subdirs to include it
    stim_loc = np.where(subdirs==stim)[0][0] #which subset is the stimulus
    stim_subdirs = subdirs[stim_loc-1:stim_loc+1] #take stim and previous 
    expt_subdirs = subdirs[np.where(subdirs!=stim)[0]] # all but stim
    
    # create disctionaries combining all mean spike shapes other than stim
    ename_neurons = {}
    ename_times = {}
    neuron_matches = []
    for subdir in expt_subdirs:
        subdir_path = result_path+'/'+subdir+'/'
        nfiles = np.sort(os.listdir(subdir_path))
        ename_neurons[subdir] = {}
        ename_times[subdir] = {}
        for nn in nfiles:
            if nn[-3:]=='npy' and nn[7]=='t' and nn[:3]==ename:
                spike_times = np.load(subdir_path+'/'+nn)
                spike_shapes = np.load(subdir_path+'/'+nn[:7]+'profiles.npy')
                ename_neurons[subdir][nn[5]] = spike_shapes.mean(0)
                ename_times[subdir][nn[5]] = spike_times
        neuron_matches+=[[subdir]]
    
    # if file is not found for any neurons on the channel, skip it
    if all([ename_neurons[subdir].items()==[] for subdir in expt_subdirs]):
        print "No neurons ID'd in experiment for "+ename
        return
    else:
        # if neurons exist, make directories to store them
        # spike shapes and times for the full neurons
        os.mkdir(result_path+'/combined_neurons/'+ename)
        # information on how the neurons were combined
        os.mkdir(result_path+'/combined_neurons/'+ename+'/combination_info')
        # plots resulting from the full neurons
        os.mkdir(result_path+'/combined_neurons/'+ename+'/full_time_plots')
        # all the information about neurons which were incomplete or didnt show
        # up at every time point
        os.mkdir(result_path+'/combined_neurons/'+ename+'/incomplete_neurons')
    
    # if neurons are only found in some of the experiment
    
    # now, compare across adjacent time points to find matches
    # neuron matches will track neuron ids that match across time
    if len(ename_neurons[expt_subdirs[0]].keys())>0:
        neuron_matches[0]+= list(np.sort(ename_neurons[expt_subdirs[0]].keys()))
    else:
         neuron_matches[0]+=['-1']
    for subdiridx,subdir in enumerate(expt_subdirs[:-1]):
        subdir1 = expt_subdirs[subdiridx+1]
        neurons_first = np.sort(ename_neurons[subdir].keys())
        neurons_second = np.sort(ename_neurons[subdir1].keys())
        # finds which are most similar
        sim_mat = np.zeros((len(neurons_first), len(neurons_second)))
        for comparison in itertools.product(neurons_first, neurons_second):
            c0 = comparison[0]; c1 = comparison[1]
            shape1 = ename_neurons[subdir][c0]
            shape2 = ename_neurons[subdir1][c1]
            #set up the similarity matrix
            sim_mat[np.where(neurons_first==c0), 
                    np.where(neurons_second==c1)]=\
                                    np.corrcoef(shape1, shape2)[0,1]
        
        # hey if there is nothing in the similarity matrix ignore it since 
        # therefore nothing is connected
        if sim_mat.shape[1]==0:
            # if nothing shows up during the stimulus, all neurons have nomatch
            neuron_matches[subdiridx+1]+=['-1']*sim_mat.shape[0] 
        elif sim_mat.shape[0]==0:
            neuron_matches[subdiridx+1]+=['-1']+list(neurons_second)
        else:
            finished = False
            last_sort = neuron_matches[subdiridx][1:]
            this_sort = -1*np.ones(len(last_sort))
            while finished is False:
                min_loc = np.unravel_index(sim_mat.argmax(), sim_mat.shape)
                if sim_mat[min_loc] > 0.9: #if the fit is reasonable...
                    # match the new neuron to its last position
                    this_sort[np.where(
                        np.asarray(last_sort)==str(neurons_first[min_loc[0]]))[0]] =\
                                                            neurons_second[min_loc[1]]
                sim_mat[:,min_loc[1]]+=-1 # add to the sim mat to denote finished
                sim_mat[min_loc[0],:]+=-1 # same idea
                # finish if all matching is done
                if (sim_mat<0).all():
                    # add back neurons that didnt fit anything
                    disconnected_neurons = list(set(neurons_second)
                                        -set(this_sort.astype(int).astype(str)))
                    finished=True
            neuron_matches[subdiridx+1]+= (
                    list(this_sort.astype(int).astype(str))+disconnected_neurons ) 
    ### The experiment is now sorted. The stimulus is now being sorted
    
    # now check stimulus and figure out which is which
    # this part is identical to above and any changes should match
    stim_neurons = {}
    stim_times = {}
    stim_matches = []
    for subdir in stim_subdirs:
        subdir_path = result_path+'/'+subdir+'/'
        nfiles = np.sort(os.listdir(subdir_path))
        stim_neurons[subdir] = {}
        stim_times[subdir] = {}
        for nn in nfiles:
            if nn[-3:]=='npy' and nn[7]=='t' and nn[:3]==ename:
                spike_times = np.load(subdir_path+'/'+nn)
                spike_shapes = np.load(subdir_path+'/'+nn[:7]+'profiles.npy')
                stim_neurons[subdir][nn[5]] = spike_shapes.mean(0)
                stim_times[subdir][nn[5]] = spike_times
        stim_matches+=[[subdir]]
    stim_matches[0]+= list(np.sort(stim_neurons[stim_subdirs[0]].keys()))
        
    # stim matching
    subdir = stim_subdirs[0]
    subdir1 = stim
    neurons_first = np.sort(stim_neurons[subdir].keys())
    neurons_second = np.sort(stim_neurons[subdir1].keys())
    sim_mat = np.zeros((len(neurons_first), len(neurons_second)))
    # finds which are most similar
    for comparison in itertools.product(neurons_first, neurons_second):
        c0 = comparison[0]; c1 = comparison[1]
        shape1 = stim_neurons[subdir][c0]
        shape2 = stim_neurons[subdir1][c1]
        #set up the similarity matrix
        sim_mat[np.where(neurons_first==c0), 
                np.where(neurons_second==c1)]=\
                                np.corrcoef(shape1, shape2)[0,1]
    
    if sim_mat.shape==(0,0):
        # nothing during the comparison period, nothing shows up during stim
        stim_matches[0]+=['-1']
        stim_matches[1]+=['-1']
    elif sim_mat.shape[1]==0:
        # nothing shows up during the stimulus
        stim_matches[1]+=['-1']*sim_mat.shape[0] # all neurons have no match
    elif sim_mat.shape[0]==0:
        #nothing shows up during related recording
        stim_matches[0]+=['-1']*sim_mat.shape[1]
        stim_matches[1]+=['-1']*sim_mat.shape[1]+list(set(neurons_second))
    else:
        finished = False
        last_sort = stim_matches[0][1:]
        this_sort = -1*np.ones(len(last_sort))
        while finished is False:
            min_loc = np.unravel_index(sim_mat.argmax(), sim_mat.shape)
            if sim_mat[min_loc] >0.9: #if the fit is reasonable...
                # match the new neuron to its last position
                this_sort[np.where(
                        np.asarray(last_sort)==str(neurons_first[min_loc[0]]))[0]] =\
                                                            neurons_second[min_loc[1]]
            sim_mat[:,min_loc[1]]+=-1 # add to the sim mat to denote finished
            sim_mat[min_loc[0],:]+=-1 # same idea
            # finish if all matching is done
            if (sim_mat<0).all():
                # add back neurons that didnt fit anything
                disconnected_neurons = list(set(neurons_second)
                                    -set(this_sort.astype(int).astype(str)))
                finished=True
        stim_matches[1]+= (
                list(this_sort.astype(int).astype(str))+disconnected_neurons ) 
            
    # now, we need to line stim_matches up with the neuron_matches
    neuron_matches+=[[stim]]
    match_line = np.where(expt_subdirs==stim_subdirs[0])[0][0]
    for neuron_ind in neuron_matches[match_line][1:]:
        if neuron_ind=='-1':
            neuron_matches[-1]+=['-1']
        else:
            # find where the stim neuron id will be located
            stim_location = np.where(np.asarray(stim_matches[0])==neuron_ind)[0][0]
            # save it to neuron_matches
            neuron_matches[-1]+=[stim_matches[1][stim_location]]
    
    # make sure we 
    disconnected_neurons = list(set(stim_matches[1])
                                -set(neuron_matches[-1]))
    neuron_matches[-1]+=disconnected_neurons
    
    # fix format of matches so the lengths are the same
    length = len(sorted(neuron_matches,key=len, reverse=True)[0])
    matching =np.array([list(xi)+[-1]*(length-len(xi)) for xi in np.asarray(neuron_matches)]).T
    matching = np.hstack([np.asarray([range(length)]).T, matching])
    header = 'neuron_id,'
    for sub in expt_subdirs:
        header+=sub+','
    header+=stim+','
    np.savetxt(result_path+'/combined_neurons/'+ename+
            '/combination_info/neuron_matching.csv', 
               matching[1:,:].astype(float), delimiter=',', header=header[:-1])
               
    # now re-assemble the data, save plots of waveforms from each time
    # go through neurons, return the experimental part
    segments = matching[0,1:-1]
    # start the plot of incomplete neurons
    incomplete_fig = plt.figure(figsize=(8,6))
    ax = plt.subplot()
    for midx, new_id in enumerate(matching[1:,0]):
        match_row = midx+1
        expt_matches = matching[match_row][1:-1]
        #the locations of the time before stimulus, and stimulus
        #stim_loc -1 (since it's the one before stim) +1 (due to the index col)
        pre_loc = stim_loc-1+1
        stim_matches = matching[match_row][[pre_loc,-1]]
        
        # get a counter going
        counter = 0
        full_times = []
        full_spikes = []
        names = ''
        # start collecting the data
        for idx, neuron_id in enumerate(expt_matches):
            # if the neuron exists
            if neuron_id!='-1':
                counter+=1
                segment = segments[idx]
                names+= segment[0]
                # collect spikes and times
                times  = np.load(result_path+'/'+segment+'/'+ename+'_n'+
                                neuron_id+'_times.npy')
                spikes = np.load(result_path+'/'+segment+'/'+ename+'_n'+
                                neuron_id+'_profiles.npy')
                full_times.append(times)
                full_spikes.append(spikes)
                
            if neuron_id=='-1' or idx+1==len(expt_matches):
                # neuron no longer exists, now figure out if it's worth keeping
                # may either be end of experiment, or just a -1
                if counter > 1: #save it as a full_neuron
      
                    # plot everyting
                    colors = 'rgbyk'
                    full_fig = plt.figure()
                    bx = plt.subplot()
                    for sidx, spikes in enumerate(full_spikes):
                        bx.plot(spikes.mean(0), color=colors[sidx%5], 
                                label = names[sidx])
                        bx.fill_between(range(40), 
                                        spikes.mean(0)+spikes.std(0), 
                                        spikes.mean(0)-spikes.std(0),
                                        color=colors[sidx%5], alpha=0.1)
                    
                    # if stimulus is tied to this full neuron AND exists
                    if (names.find(str(stim_loc-1)) >= 0 
                                and stim_matches[1]!='-1'): 
                        # true if 
                        names+='s'
                        times = np.load(result_path+'/'+stim+'/'+ename+'_n'+
                                        stim_matches[-1]+'_times.npy')
                        spikes = np.load(result_path+'/'+stim+'/'+ename+'_n'+
                                        stim_matches[-1]+'_profiles.npy')
                        bx.plot(spikes.mean(0), label = stim)
                        np.save(result_path+'/combined_neurons/'+ename+
                                '/'+ename+'_neuron'+new_id+'_'+names+
                                '_stim_times.npy', times)
                        np.save(result_path+'/combined_neurons/'+ename+
                                '/'+ename+'_neuron'+new_id+'_'+names+
                                '_stim_spikes.npy', spikes)
                        fig2 = plot_isi(np.sort(times))
                        fig2.savefig(result_path+'/combined_neurons/'+ename+
                                '/full_time_plots/'+ename+'_neuron'+new_id+'_'+names
                                +'_stim_isi.png')
                        plt.close(fig2)
                    
                    # save spikes and times
                    np.save(result_path+'/combined_neurons/'+ename+
                        '/'+ename+'_neuron'+new_id+'_'+names
                        +'_times.npy', np.hstack(full_times))
                    np.save(result_path+'/combined_neurons/'+ename+
                        '/'+ename+'_neuron'+new_id+'_'+names
                        +'_spikes.npy', np.vstack(full_spikes).T)
                    
                    # finish the plot of spike shapes now that 
                    # stim may have been added
                    bx.legend(); bx.set_ylim([-0.0001, 0.0001])
                    plt.tight_layout()
                    full_fig.savefig(result_path+'/combined_neurons/'+ename+
                        '/combination_info/'+ename+'_neuron'+new_id+'_'+names
                        +'_spikes.png')
                    plt.close(full_fig)
                    
                    # plot isi
                    fig2 = plot_isi(np.sort(np.hstack(full_times)))
                    fig2.savefig(result_path+'/combined_neurons/'+ename+
                                '/full_time_plots/'+ename+'_neuron'+new_id+'_'+names
                                +'_isi.png')
                    plt.close(fig2)
                    fig3 = plot_isi_min(np.sort(np.hstack(full_times)))
                    fig3.savefig(result_path+'/combined_neurons/'+ename+
                                '/full_time_plots/'+ename+'_neuron'+new_id+'_'+names
                                +'_refractory.png')
                    plt.close(fig3)
                        
                
                elif counter==1: # save it as an incomplete neuron
                    full_times = full_times[0]
                    full_spikes = full_spikes[0]
                    ax.plot(spikes.mean(0), label = 'n'+new_id+'_'+segment)
                    np.save(result_path+'/combined_neurons/'+ename+
                            '/incomplete_neurons/neuron'+new_id+'_'
                            +segment+'_spikes.npy', full_spikes)
                    np.save(result_path+'/combined_neurons/'+ename+
                            '/incomplete_neurons/neuron'+new_id+'_'
                            +segment+'_times.npy', full_times)
                    
                    # if somehow the stimulus is attached to a recording
                    # of a neuron only in the one day before stim
                    if (names.find(stim_matches[0]) >= 0 
                                and stim_matches[1]!='-1'): # true if 
                        times = np.load(result_path+'/'+stim+'/'+ename+'_n'+
                                        stim_matches[-1]+'_times.npy')
                        spikes = np.load(result_path+'/'+stim+'/'+ename+'_n'+
                                        stim_matches[-1]+'_profiles.npy')
                        ax.plot(spikes.mean(0), label = 'n'+new_id+'_'+stim)
                        np.save(result_path+'/combined_neurons/'+ename+
                                '/incomplete_neurons/'+ename+'_neuron'+new_id+'_'+names+
                                '_stim_times.npy', times)
                        np.save(result_path+'/combined_neurons/'+ename+
                                '/incomplete_neurons/'+ename+'_neuron'+new_id+'_'+names+
                                '_stim_spikes.npy', spikes)

                    
                #reset the counter
                counter = 0
                full_times = []
                full_spikes = []
                names = ''
        
        # if the stimulus just shows up here but it unconnected to anything
        if (stim_matches[0]=='-1' and stim_matches[1]!='-1'):
            # the stimulus is untied to the neuron
            times = np.load(result_path+'/'+stim+'/'+ename+'_n'+
                            stim_matches[-1]+'_times.npy')
            spikes = np.load(result_path+'/'+stim+'/'+ename+'_n'+
                            stim_matches[-1]+'_profiles.npy')
            ax.plot(spikes.mean(0), label = new_id+'u_'+stim)
            np.save(result_path+'/combined_neurons/'+ename+
                    '/incomplete_neurons/neuron'+new_id+'u_'
                    +'stim_times.npy', times)
            np.save(result_path+'/combined_neurons/'+ename+
                    '/incomplete_neurons/neuron'+new_id+'u_'
                    +'stim_spikes.npy', spikes)
                
    # now we finish the plotting
    plt.legend()
    ax.set_ylim([-0.0001, 0.0001])
    plt.tight_layout()
    incomplete_fig.savefig(result_path+'/combined_neurons/'+ename+
                '/incomplete_neurons/unmatched_spikes.png')
    plt.clf()
    plt.close(incomplete_fig)
            
                
def combine_neurons(ename, subdirs):
    """ takes in electrode name, subdirectories.
    
    The subdirectories should be approx. 1 day long, and every neuron should be
    present in every subdirectory. This function stitches full neurons together
    by matching across each subdirectionry 
    
    There is no stimulus
    
    The combined neurons folder then will have folders for each electrode. In 
    the folders, there will be 2 files for each neuron:
    - spike times for the neuron
    - spike shapes for the neuron 
    Also, a csv will be included for showing how each neuron matches up across
    time.
    
    """

    expt_subdirs = subdirs
    
    # create disctionaries combining all mean spike shapes other than stim
    ename_neurons = {}
    ename_times = {}
    neuron_matches = []
    for subdir in expt_subdirs:
        subdir_path = result_path+'/'+subdir+'/'
        nfiles = np.sort(os.listdir(subdir_path))
        ename_neurons[subdir] = {}
        ename_times[subdir] = {}
        for nn in nfiles:
            if nn[-3:]=='npy' and nn[7]=='t' and nn[:3]==ename:
                spike_times = np.load(subdir_path+'/'+nn)
                spike_shapes = np.load(subdir_path+'/'+nn[:7]+'profiles.npy')
                ename_neurons[subdir][nn[5]] = spike_shapes.mean(0)
                ename_times[subdir][nn[5]] = spike_times
        neuron_matches+=[[subdir]]
    
    # if file is not found for any neurons on the channel, skip it
    if all([ename_neurons[subdir].items()==[] for subdir in expt_subdirs]):
        print "No neurons ID'd in experiment for "+ename
        return
    else:
        # if neurons exist, make directories to store them
        # spike shapes and times for the full neurons
        os.mkdir(result_path+'/combined_neurons/'+ename)
        # information on how the neurons were combined
        os.mkdir(result_path+'/combined_neurons/'+ename+'/combination_info')
        # plots resulting from the full neurons
        os.mkdir(result_path+'/combined_neurons/'+ename+'/full_time_plots')
        # all the information about neurons which were incomplete or didnt show
        # up at every time point
        os.mkdir(result_path+'/combined_neurons/'+ename+'/incomplete_neurons')
    
    # if neurons are only found in some of the experiment
    
    # now, compare across adjacent time points to find matches
    # neuron matches will track neuron ids that match across time
    if len(ename_neurons[expt_subdirs[0]].keys())>0:
        neuron_matches[0]+= list(np.sort(ename_neurons[expt_subdirs[0]].keys()))
    else:
         neuron_matches[0]+=['-1']
    for subdiridx,subdir in enumerate(expt_subdirs[:-1]):
        subdir1 = expt_subdirs[subdiridx+1]
        neurons_first = np.sort(ename_neurons[subdir].keys())
        neurons_second = np.sort(ename_neurons[subdir1].keys())
        # finds which are most similar
        sim_mat = np.zeros((len(neurons_first), len(neurons_second)))
        for comparison in itertools.product(neurons_first, neurons_second):
            c0 = comparison[0]; c1 = comparison[1]
            shape1 = ename_neurons[subdir][c0]
            shape2 = ename_neurons[subdir1][c1]
            #set up the similarity matrix
            sim_mat[np.where(neurons_first==c0), 
                    np.where(neurons_second==c1)]=\
                                    np.corrcoef(shape1, shape2)[0,1]
        
        # hey if there is nothing in the similarity matrix ignore it since 
        # therefore nothing is connected
        if sim_mat.shape[1]==0:
            # if nothing shows up during the stimulus, all neurons have nomatch
            neuron_matches[subdiridx+1]+=['-1']*sim_mat.shape[0] 
        elif sim_mat.shape[0]==0:
            neuron_matches[subdiridx+1]+=['-1']+list(neurons_second)
        else:
            finished = False
            last_sort = neuron_matches[subdiridx][1:]
            this_sort = -1*np.ones(len(last_sort))
            while finished is False:
                min_loc = np.unravel_index(sim_mat.argmax(), sim_mat.shape)
                if sim_mat[min_loc] > 0.9: #if the fit is reasonable...
                    # match the new neuron to its last position
                    this_sort[np.where(
                        np.asarray(last_sort)==str(neurons_first[min_loc[0]]))[0]] =\
                                                            neurons_second[min_loc[1]]
                sim_mat[:,min_loc[1]]+=-1 # add to the sim mat to denote finished
                sim_mat[min_loc[0],:]+=-1 # same idea
                # finish if all matching is done
                if (sim_mat<0).all():
                    # add back neurons that didnt fit anything
                    disconnected_neurons = list(set(neurons_second)
                                        -set(this_sort.astype(int).astype(str)))
                    finished=True
            neuron_matches[subdiridx+1]+= (
                    list(this_sort.astype(int).astype(str))+disconnected_neurons ) 
    ### The experiment is now sorted. The stimulus is now being sorted
    
    
    # fix format of matches so the lengths are the same
    length = len(sorted(neuron_matches,key=len, reverse=True)[0])
    matching =np.array([list(xi)+[-1]*(length-len(xi)) for xi in np.asarray(neuron_matches)]).T
    matching = np.hstack([np.asarray([range(length)]).T, matching])
    header = 'neuron_id,'
    for sub in expt_subdirs:
        header+=sub+','
    np.savetxt(result_path+'/combined_neurons/'+ename+
            '/combination_info/neuron_matching.csv', 
               matching[1:,:].astype(float), delimiter=',', header=header[:-1])
               
    # now re-assemble the data, save plots of waveforms from each time
    # go through neurons, return the experimental part
    segments = matching[0,1:]
    # start the plot of incomplete neurons
    incomplete_fig = plt.figure(figsize=(8,6))
    ax = plt.subplot()
    for midx, new_id in enumerate(matching[1:,0]):
        match_row = midx+1
        expt_matches = matching[match_row][1:]
        
        # get a counter going
        counter = 0
        full_times = []
        full_spikes = []
        names = ''
        # start collecting the data
        for idx, neuron_id in enumerate(expt_matches):
            # if the neuron exists
            if neuron_id!='-1':
                counter+=1
                segment = segments[idx]
                names+= segment[0]
                # collect spikes and times
                times  = np.load(result_path+'/'+segment+'/'+ename+'_n'+
                                neuron_id+'_times.npy')
                spikes = np.load(result_path+'/'+segment+'/'+ename+'_n'+
                                neuron_id+'_profiles.npy')
                full_times.append(times)
                full_spikes.append(spikes)
                
            if neuron_id=='-1' or idx+1==len(expt_matches):
                # neuron no longer exists, now figure out if it's worth keeping
                # may either be end of experiment, or just a -1
                if counter > 1: #save it as a full_neuron
      
                    # plot everyting
                    colors = 'rgbyk'
                    full_fig = plt.figure(figsize=(6,4))
                    bx = plt.subplot()
                    for sidx, spikes in enumerate(full_spikes):
                        bx.plot(spikes.mean(0), color=colors[sidx%5], 
                                label = names[sidx])
                        bx.fill_between(range(40), 
                                        spikes.mean(0)+spikes.std(0), 
                                        spikes.mean(0)-spikes.std(0),
                                        color=colors[sidx%5], alpha=0.1)
                    
                    # save spikes and times
                    np.save(result_path+'/combined_neurons/'+ename+
                        '/'+ename+'_neuron'+new_id+'_'+names
                        +'_times.npy', np.hstack(full_times))
                    np.save(result_path+'/combined_neurons/'+ename+
                        '/'+ename+'_neuron'+new_id+'_'+names
                        +'_spikes.npy', np.vstack(full_spikes).T)
                    
                    # finish the plot of spike shapes now
                    bx.legend(); bx.set_ylim([-0.0001, 0.0001])
                    plt.tight_layout()
                    full_fig.savefig(result_path+'/combined_neurons/'+ename+
                        '/combination_info/'+ename+'_neuron'+new_id+'_'+names
                        +'_spikes.png')
                    plt.close(full_fig)
                    
                    # plot isi
                    fig2 = plot_isi(np.sort(np.hstack(full_times)))
                    fig2.savefig(result_path+'/combined_neurons/'+ename+
                                '/full_time_plots/'+ename+'_neuron'+new_id+'_'+names
                                +'_isi.png')
                    plt.close(fig2)
                    fig3 = plot_isi_min(np.sort(np.hstack(full_times)))
                    fig3.savefig(result_path+'/combined_neurons/'+ename+
                                '/full_time_plots/'+ename+'_neuron'+new_id+'_'+names
                                +'_refractory.png')
                    plt.close(fig3)
                        
                
                elif counter==1: # save it as an incomplete neuron
                    full_times = full_times[0]
                    full_spikes = full_spikes[0]
                    ax.plot(spikes.mean(0), label = 'n'+new_id+'_'+segment)
                    np.save(result_path+'/combined_neurons/'+ename+
                            '/incomplete_neurons/neuron'+new_id+'_'
                            +segment+'_spikes.npy', full_spikes)
                    np.save(result_path+'/combined_neurons/'+ename+
                            '/incomplete_neurons/neuron'+new_id+'_'
                            +segment+'_times.npy', full_times)

                    
                #reset the counter
                counter = 0
                full_times = []
                full_spikes = []
                names = ''
                
    # now we finish the incomplete plotting
    plt.legend()
    ax.set_ylim([-0.0001, 0.0001])
    plt.tight_layout()
    incomplete_fig.savefig(result_path+'/combined_neurons/'+ename+
                '/incomplete_neurons/unmatched_spikes.png')
    plt.clf()
    plt.close(incomplete_fig)
        
        
if __name__=='__main__':

    # for spike
    enames= np.sort(np.load(experiment+'numpy_database/enames.npy'))

    
    # if the folder exists, this prevents it from being overwritten
    # if you want to overwrite it, just delete it.
    if os.path.isdir(result_path+'/combined_neurons'):
        print ("Neurons have already been combined. Please confirm or delete.")
        import sys
        sys.exit()
    else: os.mkdir(result_path+'/combined_neurons')
    
    
    
    timer = Electrode.laptimer()    
    for eidx,ename in enumerate(enames):
    	if stim is not None:
            combine_neurons_stim(ename, subdirs, stim)
        else:
            combine_neurons(ename,subdirs)
        

    print "Total combination time: "+str(np.round(timer(), 2))+"."

            
            
