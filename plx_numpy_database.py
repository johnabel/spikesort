"""
Created on Sat Aug 26 2017

@author: abel

This is a pre-processing file that will convert the plx data into a numpy
database.
"""


from __future__ import division
import numpy  as np
import scipy as sp
from concurrent import futures
import os

import Electrode as ele
from OutsideUtilities.PlexFile import *

# arguments for this file
plx_locations = 'data/example/'
database_path = plx_locations # put database in same location
num_cpus = 1 # consumes about 1gb ram/cpu
verbose=True
rethreshhold = False #False is no rethreshhold, number = number of spikes before rethresh


timer = ele.laptimer() # start timing
if os.path.isdir(database_path+'/plx_conversion_database'):
    print ("Database for conversion already exists in "+database_path+
            ". Please select a new location or delete existing database.")
    import sys
    sys.exit()
else:
    print "Initializing conversion database in " +database_path+"/numpy_database."
    os.mkdir(database_path+'/plx_conversion_database')

# load up the files
print "Reading subdirectories. Please ensure these are in sequential order."
subdirectories = np.sort(os.listdir(plx_locations+'/plx'))




def convert(dirfile):
    directory,filei = dirfile
    if verbose: print 'Running for '+directory+' '+str(filei[-8:-4])
    plex = PlexFile(plx_locations+'/plx/'+directory+filei)
    os.mkdir(database_path+'plx_conversion_database/'+directory+str(filei[-8:-4]))

    plex.read_data_header()
    # enames are the electrode names, eids are the ints that identify them
    enames = []
    eids = []
    for chan in plex.chan_headers:
        enames.append(chan.Name[-2:])
        eids.append(chan.Channel)

    out_data = plex.read_wf_data()
    for idx,ename in enumerate(enames):
        eid = eids[idx]
        electrode_locs = np.where(np.asarray(out_data['channels']).astype(int)==int(ename))[0]
        times = out_data['timestamps'][electrode_locs]
        spikes = out_data['wf_values'][electrode_locs]
        #
        np.save(database_path+'plx_conversion_database/'+directory+'/'+str(filei[-8:-4])
                +'/time_'+ename+'.npy', times)
        np.save(database_path+'plx_conversion_database/'+directory+'/'+str(filei[-8:-4])
                +'/spikes_'+ename+'.npy', spikes)

    return np.max(out_data['timestamps'])





last_max = 0
for idx, directory in enumerate(subdirectories):
    # loop through the portions of the experiment
    files_in_folder = np.sort(os.listdir(plx_locations+'/plx/'+directory))
    files = []
    os.mkdir(plx_locations+'plx_conversion_database/'+directory)
    for filei in files_in_folder:
        if filei[0]=='.':
            pass
        else:
            files.append([directory+'/', filei])
    if idx is 0:
        # identify and save the names of the electrodes
        fd = PlexFile(plx_locations+'/plx/'+subdirectories[0]+'/'+files[0][1])
        fd.read_data_header()
        enames = np.sort([chan.Name[-2:] for chan in fd.chan_headers])
        np.save(database_path+'numpy_database/enames.npy', enames)

    # parallelize creation of the database
    with futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
        result = executor.map(convert, files)

    # correct times in the database
    durations = []
    for r in result:
        # track the times
        durations.append(r)


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
