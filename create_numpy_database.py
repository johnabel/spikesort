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

import Electrode as ele

# arguments for this file
mcd_locations = 'data/rishabh_selected/'
database_path = mcd_locations # put database in same location
num_cpus = 1 # consumes about 1gb ram/cpu
verbose=True
rethreshhold = False #False is no rethreshhold, number = number of spikes before rethresh


timer = ele.laptimer() # start timing
if os.path.isdir(database_path+'numpy_database'):
    print ("Database already exists in "+database_path+
            ". Please select a new location or delete existing database.")
    import sys
    sys.exit()
else:
    print "Initializing numpy database in " +database_path+"numpy_database."
    os.mkdir(database_path+'numpy_database')

# load up the files
print "Reading subdirectories. Please ensure these are in sequential order."
subdirectories = np.sort(os.listdir(mcd_locations+'/mcd'))


#create a function to generate the database
def convert(dirfile):
    directory,filei = dirfile
    if verbose: print 'Running for '+directory+' '+str(filei[-8:-4])
    fd = ns.File(mcd_locations+'/mcd/'+directory+filei)
    os.mkdir(database_path+'numpy_database/'+directory+str(filei[-8:-4]))
    data = dict() #raw recordings
    time = dict() #time from mcd file
    count= dict() #sample count
    for entity in fd.list_entities():
        segment = 3
        # only record if entity is segment type
        if entity.entity_type == segment:
            data1 = []; time1 = []; count1 = [] 
            # lists of data to attach
            # loop for items in 
            for item_idx in range(entity.item_count):
                # apppend to data, time, count
                item_info = entity.get_data(item_idx)
                data1.append(item_info[0].tolist()[0])
                time1+= [item_info[1]]     
            channelName = entity.label[24:] # channel names
            #store data with name in the dictionary
            data[channelName] = np.asarray(data1)
            time[channelName] = np.asarray(time1)
            count[channelName] = len(time[channelName])
    # re-aligns the times of the spikes
    new_running_time=0
    for name in data.keys():
        # find the last spike time of the recording
        if len(time[name])> 0: # otherwise no spikes
            if np.max(time[name]) > new_running_time:
                new_running_time = np.max(time[name])
            # save times and spike shapes
            np.save(database_path+'numpy_database/'+directory+'/'+str(filei[-8:-4])
                    +'/time_'+name+'.npy', time[name])
            np.save(database_path+'numpy_database/'+directory+'/'+str(filei[-8:-4])
                    +'/spikes_'+name+'.npy', data[name])  
    return new_running_time, count
        




last_max = 0
total_counts = {}
for idx, directory in enumerate(subdirectories):
    # loop through the portions of the experiment
    files_in_folder = np.sort(os.listdir(mcd_locations+'/mcd/'+directory))
    files = []
    os.mkdir(mcd_locations+'numpy_database/'+directory)
    for filei in files_in_folder:
        if filei[0]=='.':
            pass
        else:
            files.append([directory+'/', filei])
    if idx is 0:
        # identify and save the names of the electrodes
        fd = ns.File(mcd_locations+'/mcd/'+subdirectories[0]+'/'+files[0][1])
        enames = np.sort([entity.label[24:] for entity in fd.list_entities()])
        np.save(database_path+'numpy_database/enames.npy', enames)
        
    # parallelize creation of the database
    with futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
        result = executor.map(convert, files)

    # correct times in the database 
    durations = []
    for r in result:
        # track the times
        durations.append(r[0])
        # track the total number of spikes for each
        counts = r[1]
        for key in counts.keys():
            if key in total_counts:
                total_counts[key] = total_counts[key]+counts[key]
            else:
                total_counts[key] = counts[key]
            
    time_correction = np.hstack([[last_max],last_max+np.cumsum(durations)[:-1]])
    for i,filei in enumerate(files):
        for name in enames:
            try:
                times = np.load(database_path+'numpy_database/'+directory+'/'
                        +str(filei[1][-8:-4]) +'/time_'+name+'.npy')
                times = times + time_correction[i]
                np.save(database_path+'numpy_database/'+directory+'/'+
                        str(filei[1][-8:-4]) +'/time_'+name+'.npy', times)
            except:
                pass # if there were no spikes
    last_max = last_max+np.cumsum(durations)[-1]


# resample if counts are a certain size
if rethreshhold is not False:
    # overflow is where the keys
    overflow_keys = []
    for key in total_counts.keys():
        if total_counts[key] > rethreshhold:
    	    overflow_keys.append(key)
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
                if np.max(spikes[:,16]) > max_spike_val:
                    max_spike_val = np.max(spikes[:,16])
                    
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
                good_locs = np.where(spikes[:,16] < new_max_spike_val)[0]
                spikes_new = spikes[good_locs,:]
                times_new = times[good_locs]
                np.save(database_path+'numpy_database/'+directory+'/'
                        +str(filei[1][-8:-4]) +'/spikes_'+ename+'.npy', spikes_new)
                np.save(database_path+'numpy_database/'+directory+'/'
                        +str(filei[1][-8:-4]) +'/time_'+ename+'.npy', times_new)
        
else:
    print "No automated rethreshholding."
    pass
    
    
    
    
    

