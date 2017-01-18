
# coding: utf-8

# In[ ]:

import os
import numpy as np
import csv
import matplotlib.pyplot as plt

MEA = '102016_MEAC'

path = '/Volumes/MEA_DATA_2/102016_MEAB/numpy_neurons/combinedtimestamps/beforestim_timestamps/Circadian2days/'

timestamps = os.listdir(path)
binsize = 30 # size in minutes
daysofrecording = 2 #number of days of recording

recordingtime = 60/binsize * daysofrecording * 24

bins = np.arange(0,recordingtime,1)
bins = bins.tolist()
bins.insert(0,MEA)
recordingseconds = bins[-1]*600

with open(MEA+'2day.csv', 'w') as csvfile: #save file name
    jtkcyclefile = csv.writer(csvfile)
    jtkcyclefile.writerow(bins)
    timestamps = np.sort(timestamps)
    for i in range(len(timestamps)):
        if timestamps[i][-3:] == 'npy':
            neuron = np.load(path+timestamps[i])
        else:
            continue
        
        counts, times, runs = plt.hist(neuron, recordingtime)
        countsfreq = counts/1800
        
        if len(countsfreq) < recordingtime:
            countsfreq = countsfreq
        else:
            countsfreq = countsfreq[:recordingtime]
            
        countsfreq = countsfreq.tolist()
        length = len(countsfreq)
        while length < recordingtime:
            countsfreq.append(0)
            length = length+1

        countsfreq.insert(0,timestamps[i])
        jtkcyclefile.writerow(countsfreq)

print 'Finished'