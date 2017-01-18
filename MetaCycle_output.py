
# coding: utf-8

# In[18]:

import numpy as np
import matplotlib.pyplot as plt
import os 
import shutil
import csv

MEA = '102016_MEAB'

path = '/Volumes/MEA_DATA_2/'+MEA+'/numpy_neurons/combinedtimestamps/beforestim_timestamps/'

path2 = '/Volumes/MEA_DATA_2/'+MEA+'/numpy_neurons/combinedtimestamps/beforestim_timestamps/1dayrecording/'
if os.path.isdir(path2) == False:
    os.mkdir(path2)

path3 = '/Volumes/MEA_DATA_2/'+MEA+'/numpy_neurons/combinedtimestamps/beforestim_timestamps/2dayrecording/'
if os.path.isdir(path3) == False:
    os.mkdir(path3)
    
path4 = '/Volumes/MEA_DATA_2/'+MEA+'/numpy_neurons/combinedtimestamps/beforestim_timestamps/3dayrecording/'
if os.path.isdir(path4) == False:
    os.mkdir(path4)

pathy = '/Volumes/MEA_DATA_2/'+MEA+'/numpy_neurons/combinedtimestamps/beforestim_timestamps/Circadian/'
if os.path.isdir(pathy) == False:
	os.mkdir(pathy)
	

timestamps = os.listdir(path)
timestamps = np.sort(timestamps)

for n in range(len(timestamps)):
    
    if timestamps[n][-3:] == 'npy':
        neuron = np.load(path+timestamps[n])
        neuron = np.sort(neuron)

        if neuron[-1] > 1800:
            length = neuron[-1]-neuron[0]
            bins = length/1800

            counts, times, runs = plt.hist(neuron, int(bins))
            countsfreq = counts/1800

            hoursrecording = length/3600
			
            if hoursrecording < 36:
                pathx = path2+timestamps[n][0:12]
                np.save(pathx, countsfreq)
                plt.figure()
                plt.plot(countsfreq)
                plt.savefig(path2+timestamps[n][0:12])
                plt.close()
            elif hoursrecording < 60:
                pathx = path3+timestamps[n][0:12]
                np.save(pathx, countsfreq)
                plt.figure()
                plt.plot(countsfreq)
                plt.savefig(path3+timestamps[n][0:12])
                plt.close()
            else:
                pathx = path4+timestamps[n][0:12]
                np.save(pathx, countsfreq)
                plt.figure()
                plt.plot(countsfreq)
                plt.savefig(path4+timestamps[n][0:12])
                plt.close()
                
    else:
        continue

#saves JTKcycle file for different timelengths
timestamps2 = os.listdir(path2)
recordingtime = 24*2
bins = np.arange(0,recordingtime,1)
bins = bins.tolist()
bins.insert(0,MEA)
recordingseconds = bins[-1]*600
maxall = []

with open(MEA+'1day.csv', 'w') as csvfile:
    jtkcyclefile = csv.writer(csvfile)
    jtkcyclefile.writerow(bins)
    timestamps2 = np.sort(timestamps2)
    for i in range(len(timestamps2)):
        if timestamps2[i][-3:] == 'npy':
            counts = np.load(path2+timestamps2[i])
        else:
            continue
        
        maxcounts = max(counts)
        maxall.append([timestamps2[i], maxcounts])
        
        if len(counts) < recordingtime:
            counts = counts
        else:
            counts = counts[:recordingtime]
        counts = counts.tolist()
        length = len(counts)
        while length < recordingtime:
            counts.append(0)
            length = length+1

        counts.insert(0,timestamps2[i])
        jtkcyclefile.writerow(counts)

np.savetxt(path2+'maxall.csv', maxall, delimiter = ',', fmt='%s')

timestamps2 = os.listdir(path3)
recordingtime = 48*2
bins = np.arange(0,recordingtime,1)
bins = bins.tolist()
bins.insert(0,MEA)
recordingseconds = bins[-1]*600
maxall = []

with open(MEA+'2day.csv', 'w') as csvfile:
    jtkcyclefile = csv.writer(csvfile)
    jtkcyclefile.writerow(bins)
    timestamps2 = np.sort(timestamps2)
    for i in range(len(timestamps2)):
        if timestamps2[i][-3:] == 'npy':
            counts = np.load(path3+timestamps2[i])
        else:
            continue
            
        maxcounts = max(counts)
        maxall.append([timestamps2[i], maxcounts])
        
        if len(counts) < recordingtime:
            counts = counts
        else:
            counts = counts[:recordingtime]
        counts = counts.tolist()
        length = len(counts)
        while length < recordingtime:
            counts.append(0)
            length = length+1

        counts.insert(0,timestamps2[i])
        jtkcyclefile.writerow(counts)

np.savetxt(path3+'maxall.csv', maxall, delimiter = ',', fmt='%s')

timestamps2 = os.listdir(path4)
recordingtime = 72*2
bins = np.arange(0,recordingtime,1)
bins = bins.tolist()
bins.insert(0,MEA)
recordingseconds = bins[-1]*600
maxall = []

with open(MEA+'3day.csv', 'w') as csvfile:
    jtkcyclefile = csv.writer(csvfile)
    jtkcyclefile.writerow(bins)
    timestamps2 = np.sort(timestamps2)
    for i in range(len(timestamps2)):
        if timestamps2[i][-3:] == 'npy':
            counts = np.load(path4+timestamps2[i])
        else:
            continue
            
        maxcounts = max(counts)
        maxall.append([timestamps2[i], maxcounts])
        
        if len(counts) < recordingtime:
            counts = counts
        else:
            counts = counts[:recordingtime]
        counts = counts.tolist()
        length = len(counts)
        while length < recordingtime:
            counts.append(0)
            length = length+1

        counts.insert(0,timestamps2[i])
        jtkcyclefile.writerow(counts)

np.savetxt(path4+'maxall.csv', maxall, delimiter = ',', fmt='%s')

print 'Finished!'
                

