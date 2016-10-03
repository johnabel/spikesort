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
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt

class Multielectrode(object):
    """ Class to interpret multielectrode array data. The inputs will be
    electrode
    """
    
    def __init__(self, path, locations=None, vip_locations=None):
        """
        Setup the required information. This class will be constructed from the
        firing data of many electrodes.
        ----
        data : str
            path to folder containing multielectrode array data in mcd format
        locations : optional numpy.ndarray [x,y]
            x,y locations specifying neuronal locations on the MEA
        vip_locations : optional numpy.ndarray [x,y]
            x,y locations specifying cells which express VIP
        """
        self.path = path
        self.locations = locations
        self.vip_locations = vip_locations
        
        # load mcd to numpy array
        self.raw_data, self.time, self.count = _load_mcd(path)


    
        
            
# utility functions

def _pca(data, n_components=None):
    pca = decomposition.pca.PCA(n_components=n_components)
    pca.fit(data)
    return pca

def _load_mcd(path):
    """ Utility to load a .mcd file"""
    fd = ns.File(path)
    data = dict() #raw recordings
    time = dict() #time from mcd file
    count= dict() #sample count
    for entity in fd.list_entities():
        segment = 3
        # only record if entity is segment type
        if entity.entity_type == segment:
            data1 = []; time1 = []; count1 = [] # lists of data to attach
            # loop for items in 
            for item_idx in range(entity.item_count):
                # apppend to data, time, count
                item_info = entity.get_data(item_idx)
                data1+= item_info[0].tolist()[0]
                time1+= [item_info[1]] # change to be the actual times of sampl
                count1+= [item_info[0]]
                
            channelName = entity.label[0:4]+entity.label[23:] # channel names
            #store data with name in the dictionary
            data[channelName] = np.asarray(data1)
            time[channelName] = np.asarray(time1)
            count[channelName] = np.asarray(count1)

    #return dictionary
    return data, time, count
    

if __name__=="__main__":

    #trial data
    pass
    
    
    
    