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
database_path = experiment # put subsamples in this spot
fraction_subsample = 0.05



def resample(inputs):
    path, idx, ename, frac = inputs
    d1, t1, s1 = Electrode._load_database_subsample(path, idx, ename, frac)
    return d1, t1, s1

if __name__=='__main__':

    # for spike
    enames= np.sort(np.load(experiment+'numpy_database/enames.npy'))
    files_in_folder = np.sort(os.listdir(experiment+'numpy_database'))[:-1]
    mcd_files = []
    for filei in files_in_folder:
        if filei[0]=='.':
            pass
        else:
            mcd_files.append(filei)
    # mcd labels is to -1 so as to remove enames.npy from the list.
    
    # if the folder exists, this prevents it from being overwritten
    # if you want to overwrite it, just delete it.
    if os.path.isdir(database_path+'/subsampled_test_sets'):
        print ("Subsampled already exists in "+database_path+
                ". Please select a new location.")
        import sys
        sys.exit()
    else: os.mkdir(database_path+'/subsampled_test_sets')
    
    # section for subsampling, serial ~1.5hr total maybe more
    timer = Electrode.laptimer()
    for ename in enames:
        resample_inputs = [[experiment, label, ename, fraction_subsample] 
                                for label in mcd_labels]
        resampled_data = []
        for res_inp in resample_inputs:
            resampled_data.append(resample(res_inp))
        
        print "Resampled "+ename+"."
        try:
            # setup resampled data
            rda = np.hstack([d[0].T for d in resampled_data]).T
            tda = np.hstack(d[1] for d in resampled_data)
            np.save(experiment+'subsampled_test_sets/'+ename+'_rda.npy', rda)
            np.save(experiment+'subsampled_test_sets/'+ename+'_tda.npy', tda)
        except: 
            print "No spikes identified for "+ename +"."
            #expected for some
    print "Total resampling time: "+str(np.round(timer(), 2))+"."

            
            
