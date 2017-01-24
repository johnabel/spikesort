# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:48:41 2016

@author: abel, probably mazuski
"""

from __future__ import division

import numpy  as np
from numpy import matlib
import scipy as sp
from scipy import signal, stats
from sklearn import decomposition, cluster, mixture
import neuroshare as ns
from collections import Counter
import os
from time import time
import pdb

import matplotlib as mpl
import matplotlib.pyplot as plt

class Electrode(object):
    """ Class to interpret data from a single elctrode of an MEA. 
    This class will take a single electrode from multiple mcd data sets. It 
    will then find the firing times and spoke shapes of each neuron. The idea 
    is that this can be handled in parallel, and will be able to return the
    right data to construct a multielectrode array.    
    """
    
    def __init__(self, name, database_path=None, 
                             mcd_idxs=None, mcd_paths=None):
        """
        Setup the required information.
        ----
        name : str
            electrode name. we only process one at a time
        mcd_paths : optional list(str)
            paths, in chronological order multielectrode array data in mcd 
            format. if provided, Electrode will load the data from these paths.
            if not, you can use load_array_data
        database_path : optional list(str)
            path to the numpy database
        mcd_idxs : optional list(str)
            idexes from the database which will be loaded
        """
        self.name = name
        
        if database_path is not None:
            self.database_path = database_path
            self.mcd_idxs = mcd_idxs
            raw_data = []
            raw_spike_times = np.array([])
            spike_counts = []
            for idx in mcd_idxs:
                d, t, c = _load_database_byname(database_path, idx, name)
                raw_spike_times = np.hstack([raw_spike_times, t]) 
                # the spike is 20 pts in, at 20kHz
                raw_data.append(d)
                spike_counts.append(c)
            
            self.raw_data = np.array(np.vstack(raw_data))
            self.raw_spike_times = raw_spike_times
            self.spike_counts = spike_counts
            
            
        elif mcd_paths is not None:        
            self.mcd_paths = mcd_paths
            # load mcd to numpy array
            raw_data = []
            raw_spike_times = np.array([])
            spike_counts = []
            time_correction = 0
            for path in mcd_paths:
                d, t, c = _load_mcd_byname(path, name)
                raw_spike_times = np.hstack([raw_spike_times,
                                             t+time_correction]) 
                # the spike is 20 pts in, at 20kHz
                raw_data.append(d)
                spike_counts.append(c)
                time_correction = np.max(raw_spike_times)
                # time corrected just by adding to total
            
            self.raw_data = np.array(np.vstack(raw_data))
            self.raw_spike_times = raw_spike_times
            self.spike_counts = spike_counts
            
    
    def load_array_data(self, raw_data, spike_times):
        """ loads data from an array rather than from paths"""
        self.raw_data = raw_data
        self.raw_spike_times = spike_times
        self.spike_counts = len(spike_times)
        

            
    def fit_gmm(self, sort_data=None, sort_times=None, pca_param=None, 
                thresh='bics', covar_type='full', precalc_std=None, 
                bics_thresh=5000):
        """Method for sorting spikes. Follows general idea in "spike sorting"
        article on scholarpedia. This function collects data, performs PCA,
        and clusters the resulting top principal components using a 
        Gaussian mixture model.
        d
        """
        # this can be changed to deal with all of the data
        #sort_data=None; sort_times=None
        if sort_data is None:
            sort_data = np.copy(self.raw_data)
        if sort_times is None:
            sort_times = np.copy(self.raw_spike_times)

            
        if np.size(sort_data) < 1000:
            self.num_clusters = 1 # not enough spikes to try to separate them
            print ("Insufficient spikes to sort: "+str(len(sort_data))
                        +"spikes.")
        #elif np.size(sort_times) > 1E6:
        #    self.num_clusters = 1 # too many spikes
        #    print ("Excessive spikes for sorting, consider rethresholding: "
        #            +str(len(sort_data)) +" spikes.")
        else:
            # either fit a new pca or use the provided parameters
            if pca_param is None:
                pca = _pca(sort_data)
                num_comp =np.max([2,np.sum(pca.explained_variance_ratio_>0.1)])
                if num_comp > 3: num_comp = 3
            else:
                [pca_comp, pca_mean, pca_nsamples, pca_noisevar, 
                                                         num_comp]=pca_param
                pca = decomposition.pca.PCA(n_components=len(pca_comp))
                pca.components_ = pca_comp
                pca.mean_ = pca_mean
                pca.n_samples_ = pca_nsamples
                pca.noise_variance_ = pca_noisevar


            transformed_data = pca.transform(sort_data)
            reduced_transformed_data = transformed_data[:,:num_comp]
            if precalc_std is None:
                precalc_std = np.asarray([np.std(reduced_transformed_data[:,i]) 
                            for i in range(len(reduced_transformed_data[0,:]))])
                self.calc_standard_deviation = precalc_std
            rescale_rtd =np.asarray([
                    reduced_transformed_data[:,i]/(3*precalc_std[i])
                            for i in range(len(reduced_transformed_data[0,:]))]).T*0.5      
            # this allows us to change the fit data if we wish
            fit_data = rescale_rtd#[np.where(np.abs(rescale_rtd[:,0])<=1)]
            #fit_data = fit_data[np.where(np.abs(fit_data[:,1])<=1)] 
                
            # track Bayesian information criterion
            bics = [1000000]
            for i in range(1,8): # iteratively add more clusters
                gmm_comp=i
                gmm = mixture.GMM(n_components=i, 
                                  covariance_type=covar_type).fit(fit_data)
                bics.append(gmm.bic(fit_data))
                
                if thresh=='bics':
                    if (bics[-1]-bics[-2]) > -1*bics_thresh:
                        # if the BIC improves by less than 5%, we ignore the 
                        # additional cluster that is found. this is heuristic.
                        gmm_comp = i-1
                        break
            gmm = mixture.GMM(n_components=gmm_comp,
                              covariance_type=covar_type).fit(rescale_rtd)
            pred = gmm.predict(rescale_rtd)
            
            self.pred = pred
            self.pca = pca
            self.num_comp = num_comp
            self.pca_parameters = [pca.components_, pca.mean_, pca.n_samples_,
                                   pca.noise_variance_, num_comp]
            self.gmm = gmm
            self.gmm_parameters = [gmm.means_, gmm.covars_, gmm.weights_]
            self.num_clusters = len(gmm.means_)
            self.rescale_rtd = rescale_rtd
            self.reduced_transformed_data = reduced_transformed_data
            self.fit_data = fit_data
            self.bics = bics
    
    def sort_spikes(self, sort_data=None, sort_times=None, pca_param=None,
                    gmm_param=None, covar_type='full', method='em', 
                    precalc_std=None):
        """ 
        method : 
            'em' - expectation maximization (sorts all)
            'std' - 1 standard deviations from center of ellipse
        """
        # if unprovided, just sort all the data in the electrode
        
        if sort_data is None:
            sort_data = np.copy(self.raw_data)
        if sort_times is None:
            sort_times = np.copy(self.raw_spike_times)
        
        
        #set up the PCA
        if pca_param is None:
            pca = self.pca
            num_comp = self.num_comp
        else:
            [pca_comp, pca_mean, pca_nsamples, pca_noisevar, 
                                                     num_comp]=pca_param
            pca = decomposition.pca.PCA(n_components=len(pca_comp))
            pca.components_ = pca_comp
            pca.mean_ = pca_mean
            pca.n_samples_ = pca_nsamples
            pca.noise_variance_ = pca_noisevar
            self.pca = pca
        
        #set up the GMM
        if gmm_param is None:
            gmm = self.gmm
        else:
            gmm_means, gmm_covars, gmm_weights = gmm_param
            gmm = mixture.GMM(n_components=gmm_covars.shape[0],
                              covariance_type=covar_type)
            gmm.covars_ = gmm_covars
            gmm.weights_ = gmm_weights
            gmm.means_ = gmm_means
            self.gmm = gmm
        
        # transform the data given the pca
        transformed_data = pca.transform(sort_data)
        reduced_transformed_data = transformed_data[:,:num_comp]
        if precalc_std is None:
            precalc_std = np.asarray([np.std(reduced_transformed_data[:,i]) 
                        for i in range(len(reduced_transformed_data[0,:]))])
            self.calc_standard_deviation = precalc_std
        rescale_rtd =np.asarray([
                reduced_transformed_data[:,i]/(3*precalc_std[i])
                        for i in range(len(reduced_transformed_data[0,:]))]).T*0.5        
        
        if method=='em':
            # use expectation maximization to sort spikes
            pred = gmm.predict(rescale_rtd)
            
        elif method=='std':
            # instead within 2SD of the center of ellipses
            def contains_point(point, ellipse_param):
                # returns 1 if point within ellipsoid
                if len(point)==2:
                    [cx,cy,a,b] = ellipse_param
                    point = np.asarray(point)
                    centers = np.asarray([cx,cy])
                    sigma_inv = np.array([[a**-2,0],[0,b**-2]])
                    mahalanobis_dist2 = (point-centers).T.dot(sigma_inv).dot(point-centers)
                    if mahalanobis_dist2 < stats.chi2.ppf(0.5,2):
                        return 1
                    else: return 0
                if len(point)==3:
                    [cx,cy,cz,a,b,c] = ellipse_param
                    centers = np.asarray([cx,cy,cz])
                    point = np.asarray(point)
                    sigma_inv = np.array([[a**-2,0,0],[0,b**-2,0],[0,0,c**-2]])
                    mahalanobis_dist2 = (point-centers).T.dot(sigma_inv).dot(point-centers)
                    if mahalanobis_dist2 < stats.chi2.ppf(0.5,3):
                        return 1
                    else: return 0
                    
            ellipses = []
            for i, (mean, covar) in enumerate(zip(
                                            gmm.means_, gmm._get_covars())):
                # define the ellipse, forced diagonal
                v, w = np.linalg.eigh(covar)
                angle = 0.
                stds = np.sqrt(v)
                # ellipse is now two standard deviations
                #                ell = mpl.patches.Ellipse(mean, 2.*np.sqrt(v[0]), 
                #                         2.*np.sqrt(v[1]), 180 + angle, color='red')
                ell = np.hstack([mean, stds])
                ellipses.append(ell)
            
            rrtd_in_ellipses = []
            for point in rescale_rtd:
                containers = [contains_point(point, ell) for ell in ellipses]
                rrtd_in_ellipses.append(np.array(containers))
            rrtd_in_ellipses = np.asarray(rrtd_in_ellipses)
            
            pred = []
            for rrtdie in rrtd_in_ellipses:
                found = np.where(rrtdie>0)[0]
                if len(found)==1:
                    pred.append(int(found))
                else: pred.append(-1)
        pred = np.asarray(pred)
        
        # create the mean trajectories
        neurons = dict()
        neuron_spike_times = dict()
        for cluster in range(len(gmm.means_)):
            clust_idxs = np.where(pred==cluster)[0]
            neurons[str(cluster)] = sort_data[clust_idxs,:]
            neuron_spike_times[str(cluster)] = sort_times[clust_idxs]
        
        self.pred = pred
        self.num_clusters = len(gmm.means_)
        self.neurons = neurons
        self.neuron_spike_times = neuron_spike_times
        self.rescale_rtd = rescale_rtd
        self.reduced_transformed_data = reduced_transformed_data
    
    def recursive_fit_gmm(self, noise_free_data, noise_free_times, pca_param,
                          return_data='tree', bics_thresh=5000):
        """
        will cluster repeatedly until stable neurons, will then return within
        1SD of center of each. provide the data without noise to start.
        Returns either the sorted 'data' or the 'tree' of pca/cluster data for
        reconstructing this method.
        """
        
        self.fit_gmm(sort_data=noise_free_data, sort_times=noise_free_times,
                     pca_param = pca_param, thresh='bics', covar_type='full',
                     bics_thresh=bics_thresh)
        self.sort_spikes(sort_data=noise_free_data, 
                             sort_times=noise_free_times, method='em')
        working_neurons = self.neurons
        working_times = self.neuron_spike_times
        total_size = np.size(np.hstack(working_times.values()))
        converged=False; iteration = 1
        pca_tree = [[self.pca_parameters]]
        gmm_tree = [[self.gmm_parameters]]
        std_tree = [[self.calc_standard_deviation]]
        
        while converged is False and iteration <= 4:
            # maybe only a single split is necessary?
            #set up test
            working_electrodes = []
            working_clusters = []
            for neuron_id in np.sort(working_neurons.keys()):
                # take a set of data, fit, sort
                data = working_neurons[neuron_id]
                times = working_times[neuron_id]
                wele = Electrode(name='working')
                wele.load_array_data(data, times)

                wele.fit_gmm(thresh='bics', covar_type='full')
                wele.sort_spikes(method='em')


                # sufficiently sized clusters
                working_electrodes.append(wele)
                working_clusters.append(wele.num_clusters)
            
            if all(working_clusters == np.ones(len(working_clusters))):
                converged=True
            else:
                # we will iterate again
                iteration+=1
            # expand the set of working neurons, grow the tree
            working_neurons = {}
            working_times = {}
            pca_data = []
            gmm_data = []
            std_data = []
            neuron_id = 0
            for wele in working_electrodes:
                for wele_neuron_id in np.sort(wele.neurons.keys()):
                    working_neurons[neuron_id]=wele.neurons[wele_neuron_id]
                    working_times[neuron_id] = \
                                wele.neuron_spike_times[wele_neuron_id]
                    neuron_id+=1
                pca_data.append(wele.pca_parameters)
                gmm_data.append(wele.gmm_parameters)
                std_data.append(wele.calc_standard_deviation)
            pca_tree.append(pca_data)
            gmm_tree.append(gmm_data)
            std_tree.append(std_data)
        
        return pca_tree, gmm_tree, std_tree
        
    def recursive_sort_spikes(self, noise_free_data, noise_free_times, 
                              pca_tree, gmm_tree, std_tree, final_method='std'):
        """ 
        sorts provided spikes using the clustering criteria from trees.
        provide the data without noise to start.
        """
        depth = len(pca_tree) # numer of iterations used
        iterative_predictor = np.zeros(len(noise_free_times))
        for iteration in range(depth-1): # last iteration will sort differently
            pca_datas = pca_tree[iteration]
            gmm_datas = gmm_tree[iteration]
            std_datas = std_tree[iteration]
            #get the subclustersx
            predictor_locs = [np.where(iterative_predictor==subiteration)
                                for subiteration in range(len(pca_datas))]
            for subiteration in range(len(pca_datas)): # iterate thru clusters
                pca_data = pca_datas[subiteration]
                gmm_data = gmm_datas[subiteration]
                std_data = std_datas[subiteration]
                predictor_active = predictor_locs[subiteration]
                sd = noise_free_data[predictor_active]
                st = noise_free_times[predictor_active]
                # perform pca
                # sort the spikes using em
                if iteration==0:
                    covar_type='full'
                else:
                    covar_type='full'    
                self.sort_spikes(sort_data=sd, sort_times=st, 
                                 pca_param=pca_data, gmm_param=gmm_data, 
                                 method='em', covar_type =covar_type, precalc_std = std_data)
                current_predictor = self.pred
                iterative_predictor[predictor_active] = self.pred + 1 + \
                                                iterative_predictor.max()
            # iterative predictor was previously sclaed up to prevent numbering
            # conflicts
            iterative_predictor = iterative_predictor-iterative_predictor.min()
        
        # use the final sort to get rid of noisy spots
        iteration = depth-1
        pca_datas = pca_tree[iteration]
        gmm_datas = gmm_tree[iteration]
        std_datas = std_tree[iteration]
        #get the subclusters
        predictor_locs = [np.where(iterative_predictor==subiteration)
                            for subiteration in range(len(pca_datas))]
        for subiteration in range(len(pca_datas)): # iterate thru clusters
            pca_data = pca_datas[subiteration]
            gmm_data = gmm_datas[subiteration]
            std_data = std_datas[subiteration]
            predictor_active = predictor_locs[subiteration]
            sd = noise_free_data[predictor_active]
            st = noise_free_times[predictor_active]
            # perform pca
            # sort the spikes using em
            self.sort_spikes(sort_data=sd, sort_times=st, 
                             pca_param=pca_data, gmm_param=gmm_data, 
                             method=final_method, precalc_std = std_data)
            current_predictor = self.pred
            current_predictor[np.where(current_predictor==0)] = subiteration
            iterative_predictor[predictor_active] =current_predictor
            
        # reutrn the neurons that exist
        self.pred = iterative_predictor.astype(int)
        self.num_clusters = len(gmm_datas)
        self.neurons = {}
        self.neuron_spike_times = {}
        for neuron_id in range(self.num_clusters):
            self.neurons[str(neuron_id)] = noise_free_data[np.where(iterative_predictor==neuron_id)]
            self.neuron_spike_times[str(neuron_id)] = noise_free_times[np.where(iterative_predictor==neuron_id)]
        
        # also transform the data by the original PCA for visualization
        [pca_comp,pca_mean,pca_nsamples,pca_noisevar,num_comp] = pca_tree[0][0]
        pca = decomposition.pca.PCA(n_components=len(pca_comp))
        pca.components_ = pca_comp
        pca.mean_ = pca_mean
        pca.n_samples_ = pca_nsamples
        pca.noise_variance_ = pca_noisevar
        #transform the data by the pca
        transformed_data = pca.transform(noise_free_data)
        reduced_transformed_data = transformed_data[:,:num_comp]

        precalc_std = std_tree[0][0]
        self.calc_standard_deviation = precalc_std
        rescale_rtd =np.asarray([
                reduced_transformed_data[:,i]/(3*precalc_std[i])
                        for i in range(len(reduced_transformed_data[0,:]))]).T*0.5     
        self.pca = pca
        self.rescale_rtd = rescale_rtd
        self.reduced_transformed_data = reduced_transformed_data
            
    def recursive_sort_spikes_maxsep(self, noise_free_data, noise_free_times, 
                              pca_tree, gmm_tree, std_tree, final_method='std', precalc_std=None):
        """ 
        sorts provided spikes using the clustering criteria from trees.
        Each cluster is identified one up from its full separation to minimize
        the overlap with its nearest neighboring cluster. 
        provide the data without noise to start.
        """
        depth = len(pca_tree) # numer of iterations used
        iterative_predictor = np.zeros(len(noise_free_times))
        final_predictor = -100*np.ones(len(noise_free_times))
        intermed_predictor= -100*np.ones(len(noise_free_times))
        for iteration in range(depth): # last iteration will sort differently
            last_predictions = np.copy(iterative_predictor)
            pca_datas = pca_tree[iteration]
            gmm_datas = gmm_tree[iteration]
            std_datas = std_tree[iteration]
            #get the subclustersx
            predictor_locs = [np.where(iterative_predictor==subiteration)
                                for subiteration in range(len(pca_datas))]
            for subiteration in range(len(pca_datas)): # iterate thru clusters
                pca_data = pca_datas[subiteration]
                gmm_data = gmm_datas[subiteration]
                std_data = std_datas[subiteration]
                predictor_active = predictor_locs[subiteration]
                sd = noise_free_data[predictor_active]
                st = noise_free_times[predictor_active]
                # perform pca
                # sort the spikes using em
                if iteration==0:
                    covar_type='full'
                else:
                    covar_type='full' 
                self.sort_spikes(sort_data=sd, sort_times=st, 
                                 pca_param=pca_data, gmm_param=gmm_data, 
                                 method='em', covar_type =covar_type, precalc_std = std_data)
                iterative_predictor[predictor_active] = self.pred + 1 + \
                                                iterative_predictor.max()
            # iterative predictor was previously sclaed up to prevent numbering
            # conflicts
            iterative_predictor = iterative_predictor-iterative_predictor.min()
            
            #identify where the iterative predictor has not changed
            neuron_num = 0
            for past_clust in np.unique(last_predictions)[:-1]:
                for curr_clust in np.unique(iterative_predictor):
                    if all(np.where(iterative_predictor==curr_clust)[0] ==\
                        np.where(last_predictions==past_clust)[0]):
                            # the cluster is complete
                            # take the pca and gmm data at that last one and cut the middle
                            # of each cluster using those params
                            pca_data = pca_tree[iteration-1][0]
                            gmm_data = gmm_tree[iteration-1][0]
                            [pca_comp, pca_mean, pca_nsamples, pca_noisevar, 
                                                         num_comp]=pca_data

                            
                            
                            intermed_predictor[np.where(last_predictions==past_clust)] = past_clust
                            final_predictor[np.where(last_predictions==past_clust)] = self.pred[np.where(last_predictions==past_clust)]
                            
        # reutrn the neurons that exist
        self.sort_spikes(sort_data=noise_free_data, sort_times=noise_free_times, 
                                 pca_param=pca_data, gmm_param=gmm_data, 
                                 method='std', covar_type ='full', precalc_std = precalc_std)
        self.pred = final_predictor.astype(int)
        self.num_clusters = len(gmm_datas)
        self.neurons = {}
        self.neuron_spike_times = {}
        for neuron_id in range(self.num_clusters):
            self.neurons[str(neuron_id)] = noise_free_data[np.where(iterative_predictor==neuron_id)]
            self.neuron_spike_times[str(neuron_id)] = noise_free_times[np.where(iterative_predictor==neuron_id)]
        
        # also transform the data by the original PCA for visualization
        [pca_comp,pca_mean,pca_nsamples,pca_noisevar,num_comp] = pca_tree[0][0]
        pca = decomposition.pca.PCA(n_components=len(pca_comp))
        pca.components_ = pca_comp
        pca.mean_ = pca_mean
        pca.n_samples_ = pca_nsamples
        pca.noise_variance_ = pca_noisevar
        #transform the data by the pca
        transformed_data = pca.transform(sort_data)
        reduced_transformed_data = transformed_data[:,:num_comp]
        if precalc_std is None:
            precalc_std = np.asarray([np.std(reduced_transformed_data[:,i]) 
                        for i in range(len(reduced_transformed_data[0,:]))])
            self.calc_standard_deviation = precalc_std
        rescale_rtd =np.asarray([
                reduced_transformed_data[:,i]/(3*precalc_std[i])
                        for i in range(len(reduced_transformed_data[0,:]))]).T*0.5      
        self.pca = pca
        self.rescale_rtd = rescale_rtd
        self.reduced_transformed_data = reduced_transformed_data
            
            
    def remove_noise_cluster(self, method='mean_profile'):
        """
        Removes cluster of spikes arising due to gaussian noise. Methods:
        ---
        method : str
            center (default) - removes cluster with closest zero to center
            mean_profile - removes cluster with lowest abs value of mean prof
        """
        
        if self.num_clusters <2 :
            self.num_clusters =0
            remove_cluster = 0
            self.noise_cluster_id = remove_cluster
            self.neurons.pop(str(remove_cluster))
            self.neuron_spike_times.pop(str(remove_cluster))
            return
        
        if method=='center':
            center_dist = [np.linalg.norm(ele.gmm.means_[i]) 
                                for i in range(self.num_clusters)]
            remove_cluster = np.argmin(center_dist)
        
        if method=='mean_profile':
            profile_absmean = []
            for i in range(self.num_clusters):
                profile_absmean.append(np.abs(self.neurons[str(i)]).mean())
            remove_cluster = np.argmin(profile_absmean)
        
        # remove that cluster from the data
        self.noise_cluster_id = remove_cluster
        self.neurons.pop(str(remove_cluster))
        self.neuron_spike_times.pop(str(remove_cluster))
   
        
            
    def plot_clustering(self, pc1=0, pc2=1, return_fig=False):
        """
        Plots the clustering results for the electrode. pc1 and pc2 are the 
        principal components which are plotted.
        """
        #plotting utilities
        colors='bgrcmyw'
        predicted = self.pred
        elim_loc = np.where(self.pred<0)
        pred_loc = np.where(self.pred>=0)
        
        fig = plt.figure(figsize=(6,4))
        ax = plt.subplot()
        #make_ellipses(self.gmm, ax)
        clust_color = [colors[i%len(colors)] for i in predicted[pred_loc]]
        plt.scatter(self.rescale_rtd[elim_loc, pc1], self.rescale_rtd[elim_loc,pc2], marker='.',
                    alpha=0.1, c='k')
        plt.scatter(self.rescale_rtd[pred_loc,pc1], self.rescale_rtd[pred_loc,pc2], marker='.',
                    alpha=0.1, c=clust_color)
        ax.set_xlabel('PC'+str(pc1+1)); ax.set_ylabel('PC'+str(pc2+1));
        ax.set_title(self.name)
        if return_fig is True:
            return fig
        
    def plot_3dclustering(self, times, pc1=0, pc2=1, return_fig=False):
        """
        Plots the clustering results for the electrode. pc1 and pc2 are the 
        principal components which are plotted.
        """
        from mpl_toolkits.mplot3d import Axes3D


        #plotting utilities
        colors='bgrcmyw'
        predicted = self.pred
        elim_loc = np.where(self.pred<0)
        pred_loc = np.where(self.pred>=0)
        
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(111, projection='3d')
        #make_ellipses(self.gmm, ax)
        clust_color = [colors[i%len(colors)] for i in predicted[pred_loc]]
        ax.scatter(self.rescale_rtd[elim_loc, pc1], self.rescale_rtd[elim_loc,pc2], times[elim_loc], marker='.',
                    alpha=0.1, c='k')
        ax.scatter(self.rescale_rtd[pred_loc,pc1], self.rescale_rtd[pred_loc,pc2], times[pred_loc], marker='.',
                    alpha=0.1, c=clust_color)
        ax.set_xlabel('PC'+str(pc1+1)); ax.set_ylabel('PC'+str(pc2+1));
        ax.set_title(self.name)
        if return_fig is True:
            return fig
    
    def plot_heatmap(self, pc1=0, pc2=1, return_fig=False, return_img=True):
        if hasattr(self, 'gmm')==False:
            self.sort_spikes()   
        
        fig = plt.figure(figsize=(6,4))
        heatmap, xedges, yedges = np.histogram2d(
                self.rescale_rtd[:,pc1], self.rescale_rtd[:,pc2], bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        plt.clf()
        ax = plt.subplot()
        plt.imshow(heatmap.T, extent=extent, origin=0, aspect='auto')
        ax.set_xlabel('PC'+str(pc1+1)); ax.set_ylabel('PC'+str(pc2+1));
        ax.set_title(self.name)
        if return_fig is True:
            return fig
        elif return_img is True:
            return heatmap.T                
    def plot_mean_profile(self, return_fig=False):
        """
        Plots the mean spike profile for each cluster identified.
        """
        if hasattr(self, 'gmm')==False:
            self.sort_spikes()
        colors = 'bgrcmyw'
        fig = plt.figure(figsize=(6,4))
        ax = plt.subplot()
        means = []
        for i, key in enumerate(np.sort(self.neurons.keys())):
            means.append(self.neurons[key].mean(0))
            try:
                plt.fill_between(range(len(self.neurons[key].mean(0))), 
                                 self.neurons[key].max(0), 
                                 self.neurons[key].min(0), alpha=0.1, 
                                 color=colors[i%5])
                plt.plot(self.neurons[key].mean(0), label=key, 
                         color=colors[i%len(colors)])
            except ValueError:
                print str(i)+' has no max/min'
        plt.legend()
        ax.set_title(self.name)
        if return_fig is True:
            return fig, np.asarray(means)

        
# ============================================================================
#   UTILITY FUNCTIONS FOR LOADING DATA
# ============================================================================
def _load_mcd(path):
    """ Utility to load a .mcd file from path (str)"""
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
    
def _load_mcd_byname(path, name):
    """Utility to return only from electrode with name "name" from path 
    (str)"""
    fd = ns.File(path)
    for entity in fd.list_entities():
        segment = 3
        # only record if entity is segment type
        if entity.entity_type == segment:
            channelName = entity.label[0:4]+entity.label[23:]
            if name==channelName:
                data1 = []; count1 = []; time1 = []
                for item_idx in range(entity.item_count):
                    # apppend to data, time, count
                    item_info = entity.get_data(item_idx)
                    data1.append( item_info[0].tolist()[0])
                    time1+= [item_info[1]] # change to be the actual times of sampl
                    count1+= [item_info[0]]
                
                #store data with name in the dictionary
                data = np.asarray(data1)
                time = np.asarray(time1)
                count = np.asarray(count1)
                spike_count = len(time)
                #return dictionary
                return data, time, spike_count

def _load_mcd_subsample(path, name, frac_subsample):
    """Utility to return only from electrode with name "name" some fractional
    subsample of all electrode fires."""
    fd = ns.File(path)
    for entity in fd.list_entities():
        segment = 3
        # only record if entity is segment type
        if entity.entity_type == segment:
            channelName = entity.label[0:4]+entity.label[23:]
            if name==channelName:
                data1 = []; count1 = []; time1 = []
                if entity.item_count>0:
                    for item_idx in range(entity.item_count):
                        # apppend to data, time, count
                        item_info = entity.get_data(item_idx)
                        data1.append( item_info[0].tolist()[0])
                        time1+= [item_info[1]] # change to be the actual times 
                        count1+= [item_info[0]]
                    
                    #store data with name in the dictionary
                    deck = range(entity.item_count)
                    np.random.shuffle(deck)
                    data = np.asarray(data1)[:int(frac_subsample*len(deck)),:]
                    time = np.asarray(time1)[:int(frac_subsample*len(deck))]
                    count = np.asarray(count1)[:int(frac_subsample*len(deck))]
                    spike_count = len(time)
                else:
                    data = np.asarray(data1)
                    time = np.asarray(time1)
                    count = np.asarray(count1)
                    spike_count=len(time)
                #return dictionary
                return data, time, spike_count



def _load_database_byname(database_path, mcd_number, ename):
    """ loads a single electrode's data from the database """
    if type(mcd_number) is int:
        mcd_number = str(mcd_number)
    # only load if there are any spikes
    if os.path.isfile(database_path+'/'+mcd_number+
                        '/spikes_'+ename+'.npy'):
        data = np.load(database_path+'/'+mcd_number+
                        '/spikes_'+ename+'.npy')
        time =  np.load(database_path+'/'+mcd_number+
                        '/time_'+ename+'.npy')
        spike_count = len(time)
    # if no spikes add an empty one
    else: time = []; data = []; spike_count = 0
    return data, time, spike_count

def _load_database_subsample(database_path, mcd_number, ename, frac_subsample):
    """ loads a single electrode's data from the database """
    if type(mcd_number) is int:
        mcd_number = str(mcd_number)
    # only load if there are any spikes
    if os.path.isfile(database_path+'/'+mcd_number+
                        '/spikes_'+ename+'.npy'):
        data1 = np.load(database_path+'/'+mcd_number+
                        '/spikes_'+ename+'.npy')
        time1 =  np.load(database_path+'/'+mcd_number+
                        '/time_'+ename+'.npy')
        deck = range(len(time1))
        np.random.shuffle(deck)
        data = np.asarray(data1)[:int(frac_subsample*len(deck)),:]
        time = np.asarray(time1)[:int(frac_subsample*len(deck))]
        spike_count = len(time)
    # if no spikes add an empty one
    else:
        time = []; data = []; spike_count = 0
    return data, time, spike_count

# ============================================================================
#   UTILITY FUNCTIONS FOR PROCESSING DATA
# ============================================================================
def _pca(data, n_components=None):
    """ returns a sklearn.decomposition.pca.PCA object fit to the data """
    pca = decomposition.pca.PCA(n_components=n_components)
    pca.fit(data)
    return pca

# plot the firing rate of each neuron
def firing_rates(spike_times, win=10., tmax=None):
    """ takes spike times, converts to firing frequency with window of 'win' 
    seconds """
    if tmax is None:
        tmax = np.max(spike_times)
    frates = []
    times = np.arange(win/2,tmax,win)
    for t in times:
        frates.append(np.sum(
        (spike_times>t-win/2)*(spike_times<t+win/2)
                            )/win
                    )
    return times, np.asarray(frates) 

def periodogram(x, y, period_low=2, period_high=64, res=10, norm=True):
    """ calculate the periodogram at the specified frequencies, return
    periods, pgram. if norm = True, normalized pgram is returned """
    
    periods = np.linspace(period_low, period_high, res)
    # periods = np.logspace(np.log10(period_low), np.log10(period_high),
    #                       res)
    freqs = 2*np.pi/periods
    try: pgram = signal.lombscargle(x, y, freqs)
    # Scipy bug, will be fixed in 1.5.0
    except ZeroDivisionError: pgram = signal.lombscargle(x+1, y, freqs)
    # significance (see press 1994 numerical recipes, p576)
    if norm == True:
        var = np.var(y)
        pgram_norm = pgram/var
        significance =  1- (1-np.exp(-pgram_norm))**len(x)
        return periods, pgram_norm, significance
    else:
        return periods, pgram

def test_neuron_circadian(data, period_low=2, period_high=64, 
                           res=None, norm=True):
    """ Uses a lomb-scargle approach to test if neurons are indeed 
    circadian """
    tmax = np.max(data)
    times, frates = firing_rates(data, win=3600, tmax=tmax) # 1 hour windows
    times = times/3600 # get in hours from s
    if period_low is None:
        nyquist_freq = 1/2 #samples per hour / 2
        period_low = 1/nyquist_freq
    if period_high is None:
        period_high = 64
    if res is None:
        res = (period_high-period_low)*5 
    pgrams = np.zeros([res])        
    sigs = np.zeros([res])
    cell_pers, cell_pgram, cell_sig = \
                            periodogram(times, frates, 
                                        period_low = period_low,
                                        period_high = period_high, 
                                        res = res, norm = True)
    pgrams_inrange = cell_pgram[np.where(np.logical_and(cell_pers>18, cell_pers<32))]
    sigs_inrange = cell_sig[np.where(np.logical_and(cell_pers>18, cell_pers<32))]
    pers_inrange = cell_pers[np.where(np.logical_and(cell_pers>18, cell_pers<32))]
    sigmin = np.min(sigs_inrange)
    per_idx = np.argmin(sigs_inrange)
    period = pers_inrange[per_idx]
    if sigmin < 0.05:
        return True
    else: return False

class laptimer:
    """
    Whenever you call it, it times laps.
    """
    
    def __init__(self):
        self.time = time()

    def __call__(self):
        ret = time() - self.time
        self.time = time()
        return ret

    def __str__(self):
        return "%.3E"%self()

    def __repr__(self):
        return "%.3E"%self()

if __name__=="__main__":

    #trial data
    path1 = 'data/trial/MEACclearandMEADpedot032016_1104amstart.mcd'
    path2 = 'data/trial/MEACclearandMEADpedot032016_1104amstart0001.mcd'
    path3 = 'data/trial/MEACclearandMEADpedot032016_1104amstart0002.mcd'
    paths = [path1, path2, path3]
    
    name = 'spks 44B'
    
    # load all the data from name, sort spikes, delete noise
    ele = Electrode(name, paths)    
    ele.sort_spikes()
    
    # plot everything
    ele.plot_clustering()
    ele.plot_mean_profile()
    
    # remove cluster assoaciated with noise,
    # if we want, we can re-cluster once the noise is removed, to better
    # identify neurons. it's false for now.
    ele.remove_noise_cluster(recursive_cluster=False)
    print "Removing cluster number: "+str(ele.noise_cluster_id)
    ele.plot_mean_profile()
    
    

    
    #set up for 44B
    times, frates0 = firing_rates(ele.neuron_spike_times['0'])
    times, frates1 = firing_rates(ele.neuron_spike_times['2'])
    
    plt.figure(figsize=(3.5,2.42))
    plt.plot(times,frates0,label='neuron 0', color='r')
    plt.plot(times,frates1,label='neuron 2',color='g')
    plt.xlabel('Time (s)')
    plt.ylabel('Firing Rate (Hz)')
    plt.legend(loc='best')
    plt.tight_layout()
    
    
    
    
