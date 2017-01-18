
# coding: utf-8

#Divide timestamps before and after stimulation

import os
import numpy as np
import csv

MEA = '102016_MEAB'

path = '/Volumes/MEA_DATA_2/'+MEA+'/numpy_neurons/combinedtimestamps/'

path2 = '/Volumes/MEA_DATA_2/'+MEA+'/numpy_neurons/combinedtimestamps/beforestim_timestamps/'
if os.path.isdir(path2) == False:
    os.mkdir(path2)

path3 = '/Volumes/MEA_DATA_2/'+MEA+'/numpy_neurons/combinedtimestamps/afterstim_timestamps/'
if os.path.isdir(path3) == False:
    os.mkdir(path3)

path4 = '/Volumes/MEA_DATA_2/'+MEA+'/numpy_neurons/combinedtimestamps/stimulation_timestamps/'
if os.path.isdir(path4) == False:
    os.mkdir(path4)

timeofstimulation = 259359 #time of stim start in s
timeofstimulationend = 264380 #time of stim end in s

rawtimestamps = os.listdir(path)
rawtimestamps = np.sort(rawtimestamps)


for nn in range(len(rawtimestamps)):

	if rawtimestamps[nn][-3:] == 'npy':
		neuron = np.load(path+rawtimestamps[nn])
		neuron = np.sort(neuron)
		name = rawtimestamps[nn][1:12]
		beforestim = []
		afterstim = []
		stimulation = []
		for time in range(len(neuron)):
			if neuron[time] < timeofstimulation:
				beforestim = np.append(beforestim, neuron[time])
				
			elif neuron[time] > timeofstimulationend:
				afterstim = np.append(afterstim, neuron[time])
				
			else:
				stimulation = np.append(stimulation, neuron[time])
				
		if len(beforestim) > 1:
			np.save(path2+name+'before', beforestim)
		if len(afterstim) > 1:
			np.save(path3+name+'after', afterstim)
		if len(stimulation) > 1:
			np.save(path4+name+'stim', stimulation)
		print rawtimestamps[nn]
		
	else:
		continue
		
print 'finished'
