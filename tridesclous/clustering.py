import numpy as np
import pandas as pd
import scipy.signal
import sklearn
import sklearn.decomposition
import sklearn.cluster
import sklearn.mixture
from collections import OrderedDict

def find_clusters(features, n_clusters, method='kmeans', **kargs):
    if method == 'kmeans':
        km = sklearn.cluster.KMeans(n_clusters=n_clusters,**kargs)
        labels_ = km.fit_predict(features.values)
    elif method == 'gmm':
        gmm = sklearn.mixture.GMM(n_components=n_clusters,**kargs)
        labels_ =gmm.fit_predict(features.values)
    
    labels = pd.Series(labels_, index = features.index, name = 'label')
    return labels

    

class Clustering_:
    """
    Clustering class :
        * project waveform with PCA
        * do clustering (kmean or gmm)
        * propose method for merge and split cluster.
    """
    def __init__(self, waveforms, good_events = None):
        self.waveforms = waveforms
        if good_events is None: 
            good_events = np.ones(waveforms.shape[0], dtype = bool)
        #~ self.good_events = pd.Series(good_events, index = waveforms.index, dtype = bool)
        self.good_events = good_events
        self.labels = pd.Series(index = waveforms.index,  dtype ='int32', name = 'label')
        self.labels.loc[:]= 0
    
    def project(self, method = 'pca', n_components = 5, selection = None):
        #TODO remove peak than are out to avoid PCA polution.
        wf = self.waveforms.values
        if selection is None:
            selection = self.good_events
        
        if method=='pca':
            self._pca = sklearn.decomposition.PCA(n_components = n_components)
            
            #~ feat = self._pca.fit_transform(wf)
            self._pca.fit(wf[selection])
            feat = self._pca.transform(wf)
            
            self.features = pd.DataFrame(feat, index = self.waveforms.index, columns = ['pca{}'.format(i) for i in range(n_components)])
            
        return self.features
    
    def reset(self):
        self.cluster_labels = np.unique(self.labels.values)
        self.catalogue = {}
    
    def find_clusters(self, n_clusters,method='kmeans', order_clusters = True,**kargs):
        #~ if selection is None:
            #~ selection = self.good_events
        #~ self.labels.loc[self.labels.index[selection]] = find_clusters(self.features[selection], n_clusters, method='kmeans', **kargs)
        self.labels = find_clusters(self.features, n_clusters, method='kmeans', **kargs)
        self.labels[~self.good_events] = -1
        
        self.reset()
        if order_clusters:
            self.order_clusters()
        return self.labels
    
    def merge_cluster(self, label1, label2, order_clusters = True,):
        self.labels[self.labels==label2] = label1
        self.reset()
        if order_clusters:
            self.order_clusters()
        return self.labels
    
    def split_cluster(self, label, n, method='kmeans', order_clusters = True, **kargs):
        mask = self.labels==label
        new_label = find_clusters(self.features[mask], n, method=method, **kargs)
        new_label += max(self.cluster_labels)+1
        self.labels[mask] = new_label
        self.reset()
        if order_clusters:
            self.order_clusters()
        return self.labels
    
    def order_clusters(self):
        """
        This reorder labels from highest power to lower power.
        The higher power the smaller label.
        Negative labels are not reassigned.
        """
        cluster_labels = self.cluster_labels.copy()
        cluster_labels.sort()
        cluster_labels =  cluster_labels[cluster_labels>=0]
        powers = [ ]
        for k in cluster_labels:
            wf = self.waveforms[self.labels==k].values
            #~ power = np.sum(np.median(wf, axis=0)**2)
            power = np.sum(np.abs(np.median(wf, axis=0)))
            powers.append(power)
        sorted_labels = cluster_labels[np.argsort(powers)[::-1]]
        
        #reassign labels
        N = int(max(cluster_labels)*10)
        self.labels[self.labels>=0] += N
        for new, old in enumerate(sorted_labels+N):
            #~ self.labels[self.labels==old] = new
            self.labels[self.labels==old] = cluster_labels[new]
        self.reset()
    
    def construct_catalogue(self):
        """
        
        """
        
        self.catalogue = OrderedDict()
        nb_channel = self.waveforms.columns.levels[0].size
        for k in self.cluster_labels:
            # take peak of this cluster
            # and reshaape (nb_peak, nb_channel, nb_csample)
            wf = self.waveforms[self.labels==k].values
            wf = wf.reshape(wf.shape[0], nb_channel, -1)
            
            #compute first and second derivative on dim=2
            kernel = np.array([1,0,-1])/2.
            kernel = kernel[None, None, :]
            wfD =  scipy.signal.fftconvolve(wf,kernel,'same') # first derivative
            wfDD =  scipy.signal.fftconvolve(wfD,kernel,'same') # second derivative
            
            # medians
            center = np.median(wf, axis=0)
            centerD = np.median(wfD, axis=0)
            centerDD = np.median(wfDD, axis=0)
            mad = np.median(np.abs(wf-center),axis=0)*1.4826
            
            #eliminate margin because of border effect of derivative and reshape
            center = center[:, 2:-2]
            centerD = centerD[:, 2:-2]
            centerDD = centerDD[:, 2:-2]
            
            self.catalogue[k] = {'center' : center.reshape(-1), 'centerD' : centerD.reshape(-1), 'centerDD': centerDD.reshape(-1) }
            
            #this is for plotting pupose
            mad = mad[:, 2:-2].reshape(-1)
            self.catalogue[k]['mad'] = mad
            self.catalogue[k]['channel_peak_max'] = np.argmax(np.max(center, axis=1))

        
        return self.catalogue



from .mpl_plot import ClusteringPlot
class Clustering(Clustering_, ClusteringPlot):
    pass




