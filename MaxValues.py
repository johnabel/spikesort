
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import csv

MEA = '102016_MEAB'

pathtimestamps = '/Volumes/MEA_DATA_2/'+MEA+'/numpy_neurons/combinedtimestamps/beforestim_timestamps/Circadian3days/'
timestamps = os.listdir(pathtimestamps) # lists the directory containing all the timestamps from that MEA
timestamps = np.sort(timestamps) # so that folder ends up at the end of the list

pathtosave = '/Volumes/MEA_DATA_2/'+MEA+'/numpy_neurons/combinedtimestamps/beforestim_timestamps/3day_results/'

if os.path.isdir(pathtosave) == False:
    os.mkdir(pathtosave)

for i in range(len(timestamps)):
    if timestamps[i][0:11] == 'PeriodPhase':
        periodphase = np.genfromtxt(pathtimestamps+timestamps[i], delimiter = ',')


#Calculate Maximums and Print out max and graphs

p = 0
maxvalue = []
minvalue = []
firingvalues = []
alphavalues = []

for i in range(len(timestamps)):
    if timestamps[i][-3:] == 'npy':
        pathfile = pathtimestamps + timestamps[i]
        neuron = np.load(pathfile)
        neuron = np.sort(neuron)

        length = neuron[-1]-neuron[0]
        counts, bins = np.histogram(neuron, int(length)/600) # calculates histogram in 10min bins
        counts = np.float_(counts)/600
        if p == 10000:
            jtkphase = periodphase[i-1,1]*6
            jtkperiod = periodphase[i-1,0]*6
        else:    
            jtkphase = periodphase[i,1]*6
            jtkperiod = periodphase[i,0]*6
        
        startpoint = jtkphase - (jtkperiod/2)
        if np.isnan(jtkphase):
            print 'ERROR ' + timestamps[i]
            continue

        if len(counts) <= jtkperiod:
            print 'ERROR '+timestamps[i]
            continue

        if startpoint > 0:
            startpoint = startpoint
        else:
            startpoint = jtkphase + (jtkperiod/2)

        day1start = startpoint
        day2start = startpoint + jtkperiod
        day3start = day2start + jtkperiod
        day3end = day3start + jtkperiod

        day1max = max(counts[day1start:day2start])
        day1maxtime = np.argmax(counts[day1start:day2start])
        day1min = min(counts[day1start:day2start])
        day1mintime = np.argmin(counts[day1start:day2start])
        firingtotalday1 = np.sum(counts[day1start:day2start]*600)

        if day3start < len(counts):
            day2max = max(counts[day2start:day3start])
            day2maxtime = np.argmax(counts[day2start:day3start])
            day2min = min(counts[day2start:day3start])
            day2mintime = np.argmin(counts[day2start:day3start])
            firingtotalday2 = np.sum(counts[day2start:day3start]*600)
            aftermax1 = 'NaN'
            aftermin = 'NaN'
        elif len(counts) - day2start > jtkperiod/2:
            day2max = 'NaN'
            day2min = 'NaN'
            aftermax1 = max(counts[day2start:len(counts)])
            aftermin = min(counts[day2start:len(counts)])
            aftermax1time = np.argmax(counts[day2start:len(counts)])
        else:
            day2max = 'NaN'
            day2min = 'NaN'
            aftermax1 = 'NaN'
            aftermin = 'NaN'

        if day3end < len(counts):
            day3max = max(counts[day3start:day3end])
            day3maxtime = np.argmax(counts[day3start:day3end])
            day3min = min(counts[day3start:day3end])
            day3mintime = np.argmin(counts[day3start:day3end])
            firingtotalday3 = np.sum(counts[day3start:day3end]*600)
            aftermax2 = 'NaN'
            aftermin = 'NaN'
        elif len(counts) - day3start > jtkperiod/2:
            aftermax2 = max(counts[day3start:len(counts)])
            aftermin = min(counts[day2start:len(counts)])
            aftermax2time = np.argmax(counts[day3start:len(counts)])
            day3max = 'NaN'
            day3min = 'NaN'
        else:
            day3max = 'NaN'
            day3min = 'NaN'
            aftermax2 = 'NaN'
            aftermin = 'NaN'

        if day1start > jtkperiod/2:
            beforemax = max(counts[0:day1start])
            beforemin = min(counts[0:day1start])
            beforemaxtime = np.argmax(counts[0:day1start])
        else:
            beforemax = 'NaN'
            beforemin = 'NaN'

        maxall = [timestamps[i], day1max]
        if day2max != 'NaN':
            maxall.append(day2max)
        else:
            maxall.append(' ')
        if day3max != 'NaN':
            maxall.append(day3max)
        else: 
            maxall.append(' ')
        if beforemax != 'NaN':
            maxall.append(beforemax)
        else:
            maxall.append(' ')
        if aftermax1 != 'NaN':
            maxall.append(aftermax1)
        else:
            maxall.append(' ')
        if aftermax2 != 'NaN':
            maxall.append(aftermax2)
        else:
            maxall.append(' ')
        
        maxvalue.append(maxall)
        #print maxall
        
        minall = [timestamps[i], day1min]
        if day2min != 'NaN':
            minall.append(day2min)
        else:
        	minall.append(' ')
        if day3min != 'NaN':
            minall.append(day3min)
        else:
        	minall.append(' ')
        
        minvalue.append(minall)
        firingall = [timestamps[i], firingtotalday1]

        if day2min != 'NaN':
            firingall.append(firingtotalday2)
        else:
        	firingall.append(' ')
        if day3min != 'NaN':
            firingall.append(firingtotalday3)
        else:
        	firingall.append(' ')

        firingvalues.append(firingall)
        
        for x in range(10):
            fraction = np.float_(x+1)/np.float_(10)
            day1_fraction = (fraction*(day1max-day1min))+day1min
            if day2max != 'NaN':
                day2_fraction = (fraction*(day2max-day2min))+day2min
            else:
                day2_fraction = ' '
            if day3max != 'NaN':  
                day3_fraction = (fraction*(day3max-day3min))+day3min
            else:
                day3_fraction = ' '

            alpha = []
            alpha2 = []
            alpha3 = []

            for j in range(int(jtkperiod)):
                if (j+day1start) < len(counts):
                	if np.float((counts[j+day1start-1])) > day1_fraction:
                		alpha = np.append(alpha, 1)
                if day2max != 'NaN':
                    if np.float((counts[j+day2start-1])) > day2_fraction:
                        alpha2 = np.append(alpha2,1)
                if day3max != 'NaN':
                    if np.float((counts[j+day3start-1])) > day3_fraction:
                        alpha3 = np.append(alpha3,1)

            sumalpha1 = np.sum(alpha)
            sumalpha1h = np.float_(sumalpha1)/6
            if day2max != 'NaN':
                sumalpha2 = np.sum(alpha2)
                sumalpha2h = np.float_(sumalpha2)/6
            else:
                sumalpha2 = ' '
                sumalpha2h = ' '

            if day3max != 'NaN':
                sumalpha3 = np.sum(alpha3)
                sumalpha3h = np.float_(sumalpha3)/6
            else:
                sumalpha3 = ' '
                sumalpha3h = ' '

            sumalphatotal = [timestamps[i], fraction, sumalpha1h, sumalpha2h, sumalpha3h]
            alphavalues.append(sumalphatotal)
        
        
        #create rate histogram plot demarcating circadian days, max, min
        plt.close()
        plt.figure()
        plt.plot(counts)
        ax = plt.gca()
        ax.set_xticks([day1start,day2start,day3start,day3end], minor=False)
        ax.xaxis.grid(True, which='major', linestyle = '--')
        ax.plot([jtkphase], [counts[jtkphase]], '*', color = 'red', markersize = 20)
        ax.annotate('MaxDay1', xy = (day1maxtime+day1start, day1max), xytext=(day1maxtime+day1start, day1max+.25), arrowprops=dict(facecolor='black', width =1, headwidth = 5, shrink =0.01))
        ax.annotate('MinDay1', xy = (day1mintime+day1start, day1min), xytext=(day1mintime+day1start, day1min+.25), arrowprops=dict(facecolor='black', width =1, shrink = .01, headwidth = 5))
        
        if day2max != 'NaN':
            ax.annotate('MaxDay2', xy = (day2maxtime+day2start, day2max), xytext=(day2maxtime+day2start, day2max+.25), arrowprops=dict(facecolor='black', width =1, headwidth = 5, shrink =0.01))
            ax.annotate('MinDay2', xy = (day2mintime+day2start, day2min), xytext=(day2mintime+day2start, day2min+.25), arrowprops=dict(facecolor='black', width =1, headwidth = 5, shrink =0.01))
        if day3max != 'NaN':
            ax.annotate('MaxDay3', xy = (day3maxtime+day3start, day3max), xytext=(day3maxtime+day3start, day3max+.25), arrowprops=dict(facecolor='black', width =1, headwidth = 5, shrink =0.01))
            ax.annotate('MinDay3', xy = (day3mintime+day3start, day3min), xytext=(day3mintime+day3start, day3min+.25), arrowprops=dict(facecolor='black', width =1, headwidth = 5, shrink =0.01))
        if beforemax != 'NaN':
            ax.annotate('BeforeMax', xy = (beforemaxtime, beforemax), xytext=(beforemaxtime, beforemax+.25), arrowprops=dict(facecolor='black', width = 1, headwidth = 5, shrink =0.01))
        if aftermax1 != 'NaN':
            ax.annotate('AfterMax', xy = (aftermax1time+day2start, aftermax1), xytext=(aftermax1time+day2start, aftermax1+.25), arrowprops = dict(facecolor='black', width =1, headwidth = 5, shrink = 0.01))
        if aftermax2 != 'NaN':
            ax.annotate('AfterMax', xy = (aftermax2time+day3start, aftermax2), xytext=(aftermax2time+day3start, aftermax2+.25), arrowprops=dict(facecolor='black',width=1,headwidth=5,shrink=0.01))
        
        plt.savefig(pathtosave+timestamps[i][0:12])
    else:
        p = 10000
        continue
        
np.savetxt(pathtosave+'maxvalues.csv', maxvalue, delimiter = ',', fmt='%s')
np.savetxt(pathtosave+'minvalues.csv', minvalue, delimiter = ',', fmt='%s')
np.savetxt(pathtosave+'firingvalues.csv', firingvalues, delimiter = ',', fmt='%s')
np.savetxt(pathtosave+'alphavalues.csv', alphavalues, delimiter = ',', fmt='%s')

print 'finished!'

