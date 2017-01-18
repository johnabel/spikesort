
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os 
import shutil
import csv

MEA = '102016_MEAB'

#move timestamps from combined neurons into a single folder
path = '/Volumes/MEA_DATA_2/'+MEA+'/numpy_neurons/'
path4 = '/Volumes/MEA_DATA_2/'+MEA+'/numpy_neurons/combinedtimestamps/'
path1 = '/Volumes/MEA_DATA_2/'+MEA+'/numpy_neurons/combined_neurons/'

if os.path.isdir(path4) == False:
    os.mkdir(path4)
    
def movetimestamps(path1):
    electrodes = os.listdir(path1)
    for e in range(len(electrodes)):
        path2 = path1+electrodes[e]
        timestamps = os.listdir(path2)
        for n in range(len(timestamps)):
            if timestamps[n][-3:] == 'npy' and timestamps[n][-9] == 't':
                pathx = path2 + '/' + timestamps[n]
                pathy = path4
                shutil.move(pathx, pathy)

movetimestamps(path1)

print 'Finished!'

