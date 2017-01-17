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
experiment = 'data/stimulation_included/'
count_path = experiment # put count here



def count(inputs):
    path, idx, ename, frac = inputs
    d1, t1, s1 = Electrode._load_database_byname(path, idx, 
                                                    ename)
    return len(t1)

if __name__=='__main__':

    # for spike
    enames= np.sort(np.load(experiment+'numpy_database/enames.npy'))
    
    
    subdirectories = np.sort(os.listdir(experiment+'numpy_database'))[:-1]
    
    timer = Electrode.laptimer()  
    total_counts = []  
    for ename in enames:
        data_counts = []
        for diridx,directory in enumerate(subdirectories):
            files_in_folder = np.sort(os.listdir(experiment+'numpy_database/'
                                    +directory))
            mcd_labels = []
            for filei in files_in_folder:
                if filei[0]=='.':
                    pass
                else:
                    mcd_labels.append(filei)

            # section for subsampling, serial ~1.5hr total maybe more
        
            resample_inputs = [[experiment+'numpy_database/'+directory, label, 
                            ename] for 
                            label in mcd_labels]
            
            for res_inp in resample_inputs:
                data_counts.append(resample(res_inp))
            
        print "Counted "+ename+"."
        try:
            total_spikes = np.sum(data_counts)
            total_counts.append([ename, total_spikes])
        except: 
            print "Failed for some reason."
            
    np.savetxt(experiment+'spike_counts.csv',total_counts, delimiter=',', fmt="%s")
            
        
                
    print "Total resampling time: "+str(np.round(timer(), 2))+"."

            
            
