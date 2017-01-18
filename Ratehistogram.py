
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import csv


pathtimestamps = '/Volumes/MEA_DATA_2/102016_MEAC/numpy_neurons/combinedtimestamps/beforestim_timestamps/3dayrecording/'

path2 = '/Volumes/MEA_DATA_2/102016_MEAC/numpy_neurons/combinedtimestamps/beforestim_timestamps/3dayrecording/'
if os.path.isdir(path2) == False:
    os.mkdir(path2)


timestamps = os.listdir(pathtimestamps) # lists the directory containing all the timestamps from that MEA
timestamps = np.sort(timestamps)


for i in range(len(timestamps)):
    if timestamps[i] == '.DS_Store':
        continue
    elif timestamps[i][-3:] == 'npy':
        pathfile = pathtimestamps + timestamps[i]

        neuron = np.load(pathfile)
        neuron = np.sort(neuron)

        if neuron[-1] >600:
            length = neuron[-1]-neuron[0]
            bins = length/600

            counts, times, runs = plt.hist(neuron, int(bins))
            plt.close()     
            countsfreq = counts/600
            yaxis = times/3600
            yaxis = np.delete(yaxis, yaxis[-1])
            if len(yaxis) == len(countsfreq):

                plt.figure()
                plt.plot(yaxis, countsfreq)
                plt.title(timestamps[i])
                plt.xlabel("Hours")
                plt.ylabel("Frequency")
                plt.savefig(path2+timestamps[i][0:12])
                plt.close()

            else:
                print timestamps[i]+' error'
    else:
    	continue

print "finished!"

