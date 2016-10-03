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
mcd_locations = 'data/032016_1104amstart/'
database_path = mcd_locations # put database in same location
num_cpus = 5 # consumes about 1gb ram/cpu
verbose=True


timer = ele.laptimer() # start timing
if os.path.isdir(database_path+'/numpy_database'):
    print ("Database already exists in "+database_path+
            ". Please select a new location.")
    import sys
    sys.exit()

else: os.mkdir(database_path+'/numpy_database')

# load up the files
files = np.sort(os.listdir(mcd_locations+'/mcd'))
# correct for first file with no numbering
if files[0][-5]=='t':
    os.rename(mcd_locations+'/mcd/'+files[0], 
              mcd_locations+'/mcd/'+files[0][:-4]+'0000.mcd')
    files[0] = files[0][:-4]+'0000.mcd'

# identify and save the names of the electrodes
fd = ns.File(mcd_locations+'/mcd/'+files[0])
enames = [entity.label[24:] for entity in fd.list_entities()]
np.save(database_path+'numpy_database/enames.npy', enames)

#create a function to generate the database
def convert(filei):
    if verbose: print 'Running for '+str(filei[-8:-4])
    fd = ns.File(mcd_locations+'/mcd/'+filei)
    os.mkdir(database_path+'numpy_database/'+str(filei[-8:-4]))
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
                count1+= [item_info[0]]    
            channelName = entity.label[24:] # channel names
            #store data with name in the dictionary
            data[channelName] = np.asarray(data1)
            time[channelName] = np.asarray(time1)
            count[channelName] = np.asarray(count1)
    new_running_time=0
    for name in data.keys():
        # find the last spike time of the recording
        if len(time[name])> 0: # otherwise no spikes
            if np.max(time[name]) > new_running_time:
                new_running_time = np.max(time[name])
            # save times and spike shapes
            np.save(database_path+'numpy_database/'+str(filei[-8:-4])
                    +'/time_'+name+'.npy', time[name])
            np.save(database_path+'numpy_database/'+str(filei[-8:-4])
                    +'/spikes_'+name+'.npy', data[name])
    return new_running_time
        
# parallelize creation of the database
with futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
    result = executor.map(convert, files)

# correct times in the database 
durations = []
for r in result:
    durations.append(r)
time_correction = np.hstack([[0],np.cumsum(durations)[:-1]])
for i,filei in enumerate(files):
    for name in enames:
        try:
            times = np.load(database_path+'numpy_database/'
                    +str(filei[-8:-4]) +'/time_'+name+'.npy')
            times = times + time_correction[i]
            np.save(database_path+'numpy_database/'+str(filei[-8:-4])
                        +'/time_'+name+'.npy', times)
        except:
            pass # if there were no spikes



# return how long it took
print timer()



# comparing time to load files from db or from mcd
mcd_files = np.sort(os.listdir(mcd_locations+'mcd/'))
filei = mcd_files[0]

print "Loading time from db:  "
timer = ele.laptimer()
a1,b1,c1 = ele._load_database_subsample(mcd_locations, '0000', '21B', 0.05)
print timer()

print "Loading time from mcd: "
a,b,c = ele._load_mcd_subsample(mcd_locations+'mcd/'+filei, 'spks 21B',0.05)
print timer()
