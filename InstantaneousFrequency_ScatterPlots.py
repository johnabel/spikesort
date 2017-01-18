
# coding: utf-8

# In[1]:

import os
import numpy as np
import matplotlib.pyplot as plt
import csv
MEA = '102016_MEAB'

pathtimestamps = '/Volumes/MEA_DATA_2/'+MEA+'/numpy_neurons/combinedtimestamps/beforestim_timestamps/Circadian3days/'
path2 = '/Volumes/MEA_DATA_2/'+MEA+'/numpy_neurons/combinedtimestamps/beforestim_timestamps/3day_instantfreq/'

if os.path.isdir(path2) == False:
    os.mkdir(path2)

timestamps = os.listdir(pathtimestamps) # lists the directory containing all the timestamps from that MEA
timestamps = np.sort(timestamps)


# In[ ]:

for i in range(len(timestamps)):
    
    if timestamps[i][-3:] == 'npy':
        pathfile = pathtimestamps + timestamps[i]
        neuron = np.load(pathfile)
        
        isi = []
        freq = []

        for j in range(len(neuron)-1):
            isi.append(neuron[j+1] - neuron[j])
            freq.append(1/isi[j])

        plt.close()
        
        endtime = neuron[-1]
        neuron = np.delete(neuron, [0])
        fig, ax = plt.subplots(figsize = (15,10))
        plt.tight_layout()
        plt.scatter(neuron, freq, s=.5, color = (['blue', 'black']))
        ax.set_ylim([0,15])
        ax.set_xlim([0,endtime])
        filelable = timestamps[i][0:11]
        plt.savefig(path2+filelable)
        
        plt.close()
        fig, ax = plt.subplots(figsize = (15,10))
        plt.tight_layout()
        plt.scatter(neuron, isi, s=.5, color = (['blue', 'black']))
        ax.set_ylim([0,1])
        ax.set_xlim([0,endtime])
        filelable = timestamps[i][0:11]
        plt.savefig(path2+filelable+'isi')
    else:
        continue

print 'Finished!'


# In[ ]:



