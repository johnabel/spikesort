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
experiment = 'data/example/'
database_path = experiment # put subsamples in this spot
fraction_subsample = [1, 1]



def resample(inputs):
    path, idx, ename, frac = inputs
    d1, t1, s1 = Electrode._load_database_subsample(path, idx, 
                                                    ename, frac)
    return d1, t1, s1

if __name__=='__main__':

    # for spike
    enames= np.sort(np.load(experiment+'numpy_database/enames.npy'))
    
    # if the folder exists, this prevents it from being overwritten
    # if you want to overwrite it, just delete it.
    if os.path.isdir(database_path+'/subsampled_test_sets'):
        print ("Subsampled already exists in "+database_path+
                ". Please select a new location.")
        import sys
        sys.exit()
    else: os.mkdir(database_path+'/subsampled_test_sets')
    
    subdirectories = np.sort(os.listdir(experiment+'numpy_database'))[:-1]
    
    timer = Electrode.laptimer()    
    for diridx,directory in enumerate(subdirectories):
        files_in_folder = np.sort(os.listdir(experiment+'numpy_database/'
                                    +directory))
        os.mkdir(experiment+'subsampled_test_sets/'+directory)
        mcd_labels = []
        for filei in files_in_folder:
            if filei[0]=='.':
                pass
            else:
                mcd_labels.append(filei)

        # section for subsampling, serial ~1.5hr total maybe more
        for ename in enames:
            resample_inputs = [[experiment+'numpy_database/'+directory, label, 
                            ename, fraction_subsample[diridx]] for 
                            label in mcd_labels]
            resampled_data = []
            for res_inp in resample_inputs:
                resampled_data.append(resample(res_inp))
            
            print "Resampled "+ename+"."
            try:
                # setup resampled data
                rda = np.hstack([d[0].T for d in resampled_data]).T
                tda = np.hstack(d[1] for d in resampled_data)
                np.save(experiment+'subsampled_test_sets/'+directory+'/'
                            +ename+'_rda.npy', rda)
                np.save(experiment+'subsampled_test_sets/'+directory+'/'
                            +ename+'_tda.npy', tda)
            except: 
                print "No spikes identified for "+ename +"."
                #expected for some
    print "Total resampling time: "+str(np.round(timer(), 2))+"."

            
            
