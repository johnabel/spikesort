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
from fastdtw import fastdtw


import matplotlib as mpl
import matplotlib.pyplot as plt


#trial data
experiment = 'data/stimulation_included/'
database_path = experiment # put subsamples in this spot
result_path = experiment+"numpy_neurons"
subdirectories = np.sort(os.listdir(experiment+'numpy_database'))[:-1]


def combine_neurons(ename, subdirs, stim):
    """ takes in electrode name, subdirectories, and the subdirectory that 
    contains the stimulus"""
    
    # create disctionaries combining all mean spike shapes
    ename_neurons = {}
    ename_times = {}
    neuron_matches = []
    for subdir in subdirs:
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
    if all([ename_neurons[subdir].items()==[] for subdir in subdirs]):
        return
    
    # now, compare across adjacent time points to find matches
    # neuron matches will track neuron ids that match across time
    neuron_matches[0]+= list(np.sort(ename_neurons[subdirs[0]].keys()))
    for subdiridx,subdir in enumerate(subdirs[:-1]):
        neurons_first = np.sort(ename_neurons[subdir].keys())
        neurons_second = np.sort(ename_neurons[subdirs[subdiridx+1]].keys())
        # finds which are most similar
        sim_mat = np.zeros((len(neurons_first), len(neurons_second)))
        for comparison in itertools.product(neurons_first, neurons_second):
            shape1 = ename_neurons[subdir][comparison[0]]
            shape2 = ename_neurons[subdirs[subdiridx+1]][comparison[1]]
            sim_mat[comparison[0],comparison[1]]= fastdtw(shape1, shape2)[0]
        
        finished = False
        last_sort = neuron_matches[subdiridx][1:]
        this_sort = -1*np.ones(len(last_sort))
        while finished is False:
            min_loc = np.unravel_index(sim_mat.argmin(), sim_mat.shape)
            sim_mat[:,min_loc[1]]+=1 # only once neuron is connected
            sim_mat[min_loc[0],:]+=1
            if sim_mat[min_loc] < 1E-4: #if the fit is reasonable...
                # match the new neuron to its last position
                this_sort[np.where(np.asarray(last_sort)==str(min_loc[0]))[0]] = str(min_loc[1])
            # finish if all matching is done
            if all(sim_mat>1):
                # add back neurons that didnt fit anything
                disconnected_neurons = list(set(neurons_second)-set(this_sort.astype(int).astype(str)))
                finished=True
        neuron_matches[subdiridx+1]+= list(this_sort.astype(int).astype(str))+disconnected_neurons   
        
    # fix format of matches so the lengths are the same
    length = len(sorted(neuron_matches,key=len, reverse=True)[0])
    matching =np.array([list(xi)+[-1]*(length-len(xi)) for xi in np.asarray(neuron_matches)]).T
    header = ''
    for sub in subdirectories:
        header+=sub+','
    np.savetxt(result_path+'/combined_neurons/'+ename+
            '/combination_info/neuron_matching.csv', 
               matching[1:,:].astype(float), delimiter=',', header=header[:-1])
               
    # now re-assemble the data, save plots of waveforms from each time
    
    return 

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
        os.mkdir(result_path+'/combined_neurons/'+ename)
        os.mkdir(result_path+'/combined_neurons/'+ename+'/combination_info')
        combine_neurons(ename, subdirectories, 1)
        

    print "Total combination time: "+str(np.round(timer(), 2))+"."

            
            
