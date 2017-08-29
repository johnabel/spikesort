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

plx_file = plx_locations+'plx/demo.plx'

example_plx =PlexFile(plx_file)






self = example_plx
self.read_data_header()
# steps
# take first plx file, create channels in the db based on channels in chan_headers
# for each plx file

for chan in self.chan_headers:
    print chan.Name


# get the waveforms, channels, and names
data_out = example_plx.read_wf_data()


# save data out to a plex database
save_to_db(data_out)


# reform the database correctly
reform_numpy_db(plx_locations)