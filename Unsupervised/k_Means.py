#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 14:11:20 2024

@author: rtessier
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn import cluster, metrics
import sys



def computeKMeans (datanp, k_min, k_max, metric):
    """
    datanp : (n_samples, n_features)
    
    search the best k for clustering with K-Means,with k between k_min and k_max

    metric = "silhouette" or "Davies-Bouldin" or "Calinski-Harabasz"
    
    return best_model, best_k, best_score
    """
    #init minimal score for each metrics 
    best_score = 0
    if (metric == "silhouette"):
        best_score = -1
    elif (metric == "Davies-Bouldin"):
        best_score = sys.maxsize
    elif (metric == "Calinski-Harabasz"):
        best_score = 0
    else :
        return -1
        
    best_k = -1
    best_model = 0
    
    
    for k in range(k_min,k_max+1):
        #fit model for k cluster
        model = cluster.KMeans(n_clusters =k ,n_init = 10, init = 'k-means++')
        model.fit(datanp)
        labels = model.labels_
        
        #compute metrics and compare with previous iteration
        if (metric == "silhouette"):
            score = metrics.silhouette_score(datanp, labels)
            if (score > best_score):
                best_score = score
                best_k = k
                best_model = model
        elif (metric == "Davies-Bouldin"):
            score = metrics.davies_bouldin_score(datanp, labels)
            if (score < best_score):
                best_score = score
                best_k = k
                best_model = model
        elif (metric == "Calinski-Harabasz"):
            score = metrics.calinski_harabasz_score(datanp, labels)
            if (score > best_score):
                best_score = score
                best_k = k
                best_model = model
        
    
    f0 = [element[0] for element in datanp]
    f1 = [element[1] for element in datanp]
    plt.scatter ( f0 , f1 , c = best_model.labels_ , s = 8 )
    plt.title ( " Data after clustering Kmeans with k = " + str(best_k) + " and metrics " + metric)
    plt.show ()
    print ("best number of clusters with Kmeans and metrics", metric, ":", best_k)
    print ("score =", best_score)
    return best_model, best_k, best_score



if __name__ == "__main__":
    path = './artificial/'
    databrut = arff.loadarff ( open ( path + "xclara.arff" , 'r') )
    datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in databrut [ 0 ] ]
    f0 = [element[0] for element in datanp]
    f1 = [element[1] for element in datanp]
    plt.scatter( f0 , f1 , s = 8 )
    plt.title( " Initial Datas" )
    plt.show()
    computeKMeans(datanp, 2, 20, "Davies-Bouldin")

    
    

