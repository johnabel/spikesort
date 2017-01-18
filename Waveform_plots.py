
# coding: utf-8

# In[6]:
import numpy as np
import matplotlib.pyplot as plt
import os    
     
#import csv files with waveforms: change # as needed for electrode
def plotwaveforms(electrode):

    path = '/Volumes/MEA_DATA/032016_MEAA/beforestim/sorting_results/plots/ %s_waveform.csv' % electrode
    path2 = '/Volumes/MEA_DATA/032016_MEAA/stimulation/sorting_results/plots/ %s_waveform.csv' % electrode
    
    if os.path.isfile(path) == True:
        beforestim = np.genfromtxt(path, delimiter = ',')
    else:
        beforestim = []
        print "Beforestim file does not exist"
    
    if os.path.isfile(path2) == True:
        stimulation = np.genfromtxt(path2, delimiter = ',')
    else:
        stimulation = []
        print "Stimulation file does not exist"
    
    xvalue = np.arange(1,41,1)
    waveformbefore = len(beforestim)
    waveformstim = len(stimulation)

    plt.subplot(3,1,1)
    color_idx = np.linspace(0, 1, len(beforestim))
    if waveformbefore == 40:
        plt.plot(xvalue,beforestim)
    else:
        for i in range(len(beforestim)):
            plt.plot(xvalue, beforestim[i], color = plt.cm.prism(color_idx[i]))
        
    plt.title('Beforestim Waveforms')
    plt.legend(np.arange(waveformbefore), fontsize = 'x-small')#, waveformstim, fonsize = 'x-small')
    axes = plt.gca()
    ylim = axes.get_ylim()

    plt.subplot(3,1,2)
    color_idx2 = np.linspace(0,1,len(stimulation))
    if waveformstim == 40:
        plt.plot(xvalue,stimulation)
    else:
    
        for i in range(len(stimulation)):
            plt.plot(xvalue, stimulation[i], color = plt.cm.jet(color_idx2[i]))
    plt.title('Stimulation Waveforms')
    axes = plt.gca()
    axes.set_ylim(ylim)

    plt.subplot(3,1,3)


    if os.path.isfile(path) == True and os.path.isfile(path2) == True:
        if len(stimulation) == 40 and len(beforestim) == 40:
            stimulationvalue = 1
            beforestimvalue = 1
        
        elif len(stimulation) == 40 and len(beforestim) != 40:
            beforestimvalue = len(beforestim)
            stimulationvalue = 1
        elif len(stimulation) != 40 and len(beforestim) == 40:
            beforestimvalue = 1
            stimulationvalue = len(stimulation)
        else:
            beforestimvalue = len(beforestim)
            stimulationvalue = len(stimulation)
    
        color_idx3 = np.linspace(0,1,beforestimvalue+stimulationvalue)
    
        for i in range(beforestimvalue+stimulationvalue):
            if i < beforestimvalue and waveformbefore != 40:
                plt.plot(xvalue, beforestim[i], color = plt.cm.jet(color_idx3[i]), lw=2)
            elif i < beforestimvalue and waveformbefore == 40:
                plt.plot(xvalue, beforestim, color = plt.cm.jet(color_idx3[i]), lw =2)
            elif i >= beforestimvalue and len(stimulation) == 40:
                plt.plot(xvalue, stimulation, '--', color = plt.cm.jet(color_idx3[i]), lw=2)
            else:
                plt.plot(xvalue, stimulation[i-waveformbefore], '--', color = plt.cm.jet(color_idx3[i]), lw=2)
        plt.title('Combined')
        waveformbefore2 = np.arange(0,beforestimvalue,1)
        waveformstim2 = np.arange(0,stimulationvalue,1)
        waveformlabel = np.append([waveformbefore2], [waveformstim2])
        plt.legend(waveformlabel, fontsize = 'x-small')#, waveformstim, fonsize = 'x-small')
    

    return beforestimvalue
    plt.show()
 
def skipneuron(electrode, skipneurons):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    path = '/Volumes/MEA_DATA/032016_MEAA/beforestim/sorting_results/plots/ %s_waveform.csv' % electrode
    path2 = '/Volumes/MEA_DATA/032016_MEAA/stimulation/sorting_results/plots/ %s_waveform.csv' % electrode

    if os.path.isfile(path) == True:
        beforestim = np.genfromtxt(path, delimiter = ',')
    else:
        beforestim = []
        print "Beforestim file does not exist"
    
    if os.path.isfile(path2) == True:
        stimulation = np.genfromtxt(path2, delimiter = ',')
    else:
        stimulation = []
        print "Stimulation file does not exist"
    
    xvalue = np.arange(1,41,1)
    waveformbefore = len(beforestim)
    
    if os.path.isfile(path) == True and os.path.isfile(path2) == True:
        if len(stimulation) == 40 and len(beforestim) == 40:
            stimulationvalue = 1
            beforestimvalue = 1
        
        elif len(stimulation) == 40 and len(beforestim) != 40:
            beforestimvalue = len(beforestim)
            stimulationvalue = 1
        elif len(stimulation) != 40 and len(beforestim) == 40:
            beforestimvalue = 1
            stimulationvalue = len(stimulation)
        else:
            beforestimvalue = len(beforestim)
            stimulationvalue = len(stimulation)
    
        color_idx3 = np.linspace(0,1,beforestimvalue+stimulationvalue)
     
    
    for i in range(beforestimvalue+stimulationvalue):
        if i not in skipneurons:
            if i < beforestimvalue and waveformbefore != 40:
                plt.plot(xvalue, beforestim[i], color = plt.cm.Set1(color_idx3[i]), lw=2)
            elif i < beforestimvalue and waveformbefore == 40:
                plt.plot(xvalue, beforestim, color = plt.cm.Set1(color_idx3[i]), lw=2)
            elif i >= beforestimvalue and len(stimulation) == 40:
                plt.plot(xvalue, stimulation, '--', color = plt.cm.Set1(color_idx3[i]), lw=2)
            else:
                plt.plot(xvalue, stimulation[i-waveformbefore], '--', color = plt.cm.Set1(color_idx3[i]), lw=2)
        plt.title('Combined')
        waveformbefore2 = np.arange(0,beforestimvalue,1)
        waveformstim2 = np.arange(0,stimulationvalue,1)
        waveformlabel = np.append([waveformbefore2], [waveformstim2])
        waveformlabel2 = np.delete(waveformlabel, skipneurons)
        plt.legend(waveformlabel2, fontsize = 'small')

    
    plt.show()




# In[3]:

