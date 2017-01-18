
# coding: utf-8

#run ISIcheck after sorting - before neuron combination
#need to remove **_profiles.npy and **_times.npy neurons that fail ISIcheck
#before running neuron combination
#code second part

import numpy as np
import matplotlib.pyplot as plt
import os 
import shutil

MEA = '072016_MEA2' # write MEA in
path = '/Volumes/MEA_DATA_2/'+MEA+'/numpy_neurons/3stim/' #make path for stimulation

def ISIcheck(MEA):
    pathtimestamps = '/Volumes/MEA_DATA_2/'+MEA+'/numpy_neurons/'
    days = os.listdir(pathtimestamps)

    completedays = ['0','1','2','4','5','6']

    for i in range(len(days)):
        if days[i][0] == completedays[0] or completedays[1] or completedays[2] or completedays[3] or completedays[4] or completedays[5]:
            path3 = pathtimestamps+days[i]+'/ISIgraphs'
            

            if os.path.isdir(pathtimestamps+days[i]+'/badISI') == False:
                os.mkdir(pathtimestamps+days[i]+'/badISI')
                
            path2 = pathtimestamps+days[i]
            timestamps = os.listdir(path2)
            for j in range(len(timestamps)):
                if timestamps[j][-7:-1] == 'mes.np':
                    neuron = np.load(path2+'/'+timestamps[j])
                    if np.sum(neuron) > 0:
                        isitimestamps = []
                        counts = []

                        for b in range(len(neuron)-1):
                            isitimestamps.append(neuron[b+1]-neuron[b])

                        bins = np.linspace(0,1,200)
                        counts, testbins = np.histogram(isitimestamps, bins)
                        
                        if counts[0] > 0 and np.sum(counts) > 0:
                            fractionbadspikes = counts[0]/np.sum(counts)
                        else:
                            fractionbadspikes = 0

                        if fractionbadspikes > 0.005 or counts[0] > 500:
                            pathx = path2+'/'+timestamps[j]
                            pathy = path2+'/badISI'
                            shutil.move(pathx, pathy)
                            
                            z = len(timestamps[j])-9
                            secondfile = timestamps[j][0:z]+'profiles.npy'
                            
                            pathz = path2+'/'+secondfile
                            pathy = path2+'/badISI'
                            shutil.move(pathz, pathy)
                            j = j-2
                            if j < 0:
                                j = 0

                            print 'day', i, timestamps[j]

def ISIcalculate(neuron):
    isitimestamps = []
    counts = []
    
    for b in range(len(neuron)-1):
        isitimestamps.append(neuron[b+1]-neuron[b])
    
    bins = np.linspace(0,1,200)
    counts, testbins = np.histogram(isitimestamps, bins)
    
    return counts, testbins

# calculates ISIs for stimulation file

timestamps = os.listdir(path)
if os.path.isdir(path+'ISIgraphs') == False:
    os.mkdir(path+'ISIgraphs')

for i in range(len(timestamps)):
    if timestamps[i][-7:-1] == 'mes.np':
        neuron = np.load(path+timestamps[i])
        
        counts, testbins = ISIcalculate(neuron)
        
        plt.close()
        binplot = np.delete(testbins, [199], None)
        
        plt.figure()
        plt.plot(binplot, counts)
        plt.title(timestamps[i])
        plt.xlabel("Time (s)")
        plt.ylabel("# of spikes")
        plt.savefig(path+'ISIgraphs/'+timestamps[i][0:12])

ISIcheck(MEA) # calculates bad ISIs for MEA

print 'Finished!'
