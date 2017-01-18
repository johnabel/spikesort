# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:48:41 2016

@author: abel, probably mazuski

This is a pre-processing file that will convert the data into a pickled object.
"""


from __future__ import division
import numpy  as np
import scipy as sp
import cPickle as pickle
from concurrent import futures
import neuroshare as ns
import os


# arguments for this file
mcd_locations = '/Volumes/MEA_DATA_2/102016_MEAC/'
database_path = mcd_locations # put database in same location
num_cpus = 1 # consumes about 1gb ram/cpu
rethreshhold = 10000000


# load up the files
print "Reading subdirectories. Please ensure these are in sequential order."
subdirectories = np.sort(os.listdir(mcd_locations+'/mcd'))

overflow_keys = [' 46', ' 52', ' 56', ' 58', ' 68', ' 76', ' 84', ' 87']

# resample if counts are a certain size
if rethreshhold is not False:

    print 'Rethreshholding at 125% for: '
    print overflow_keys
    for ename in overflow_keys:
        # rethreshhold these
        # first, find the max spike value
        max_spike_val = -1
        for idx, directory in enumerate(subdirectories):
            # what files are in each part
            files_in_folder = np.sort(os.listdir(mcd_locations+'/mcd/'+directory))
            files = []
            for filei in files_in_folder:
                if filei[0]=='.':
                    pass
                else:
                    files.append([directory+'/', filei])
                    
            for i,filei in enumerate(files):
                spikes = np.load(database_path+'numpy_database/'+directory+'/'
                        +str(filei[1][-8:-4]) +'/spikes_'+ename+'.npy')
                if np.max(spikes[:,20]) > max_spike_val:
                    max_spike_val = np.max(spikes[:,20])
                    
        # set a new threshhold
        new_max_spike_val = 1.25*max_spike_val
        
        # apply it to every piece of data...
        for idx, directory in enumerate(subdirectories):
            # what files are in each part
            files_in_folder = np.sort(os.listdir(mcd_locations+'/mcd/'+directory))
            files = []
            for filei in files_in_folder:
                if filei[0]=='.':
                    pass
                else:
                    files.append([directory+'/', filei])
                    
            # load it, apply it, save it
            for i,filei in enumerate(files):
                spikes = np.load(database_path+'numpy_database/'+directory+'/'
                        +str(filei[1][-8:-4]) +'/spikes_'+ename+'.npy')
                times = np.load(database_path+'numpy_database/'+directory+'/'
                        +str(filei[1][-8:-4]) +'/time_'+ename+'.npy')
                good_locs = np.where(spikes[:,20] < new_max_spike_val)[0]
                spikes_new = spikes[good_locs,:]
                times_new = times[good_locs]
                np.save(database_path+'numpy_database/'+directory+'/'
                        +str(filei[1][-8:-4]) +'/spikes_'+ename+'.npy', spikes_new)
                np.save(database_path+'numpy_database/'+directory+'/'
                        +str(filei[1][-8:-4]) +'/time_'+ename+'.npy', times_new)
        
else:
    print "No automated rethreshholding."
    pass
    
    
    
    
    

