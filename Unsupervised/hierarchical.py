#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 15:31:21 2024

@author: raphael tessier and nicolas siard
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn import cluster, metrics
import sys



def computeAglomerative (datanp, limit, k_min, k_max, threshold_min, threshold_step, threshold_max, linkage, metric):
    """
    datanp : (n_samples, n_features)
    
    limite : "threshold" or "clusters", if threshold, k arguments will be ignore; if clusters, threshold arguments will be ignore
    
    search the best k or threshold for Alglomerative clustering

    linkage :  'ward', 'complete', 'average', 'single'     

    metric : "silhouette" or "Davies-Bouldin" or "Calinski-Harabasz"
    
    return best_model, best threshold or k, best_score
    """
    #init minimal score for each metrics 
    best_score = 0
    if (metric == "silhouette"):
        best_score = -1
    elif (metric == "Davies-Bouldin"):
        best_score = 1000000
    elif (metric == "Calinski-Harabasz"):
        best_score = 0
    else :
        return -1
        
    best_k = -1
    best_t = -1
    best_model = 0

    
    
    if (limit == "clusters"):
        l_k = []
        l_score =[]  
        for k in range(k_min,k_max+1):
            model = cluster.AgglomerativeClustering(linkage = linkage , n_clusters = k )
            model = model.fit(datanp)
            labels = model.labels_
            
            #compute metrics and compare with previous iteration
            try :
                score = 0
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
                l_k.append(k)
                l_score.append(score)

            except ValueError:
                #print(k, " cluster(s) result in only on label, remove it")
                None

                    
        print ("best number of clusters with" , linkage, "aglomerative clustering and metrics", metric, ":", best_k)
        plt.xticks(l_k)
        plt.plot(l_k,l_score)
        plt.title ( " Score evolution for aglomerative clustering with number of cluster parameters with metrics " + metric)
        plt.show ()
        ret = best_k
                    
                    
    elif (limit == "threshold"):
        threshold = threshold_min
        l_threshold = []
        l_score =[]
        while (threshold <= threshold_max):
            model = cluster.AgglomerativeClustering(distance_threshold = threshold, linkage = linkage , n_clusters = None )
            model = model.fit(datanp)
            labels = model.labels_
            
            try:
                score = 0
                #compute metrics and compare with previous iteration
                if (metric == "silhouette"):
                    score = metrics.silhouette_score(datanp, labels)
                    if (score > best_score):
                        best_score = score
                        best_t = threshold
                        best_model = model
                elif (metric == "Davies-Bouldin"):
                    score = metrics.davies_bouldin_score(datanp, labels)
                    if (score < best_score):
                        best_score = score
                        best_t = threshold
                        best_model = model
                elif (metric == "Calinski-Harabasz"):
                    score = metrics.calinski_harabasz_score(datanp, labels)
                    if (score > best_score):
                        best_score = score
                        best_t = threshold
                        best_model = model
                l_threshold.append(threshold)
                l_score.append(score)
            
            except ValueError:
                #print(threshold, "for threshold result in only on label, remove it")
                None
                
            threshold += threshold_step
        print ("best threshold with", linkage, "aglomerative clustering and metrics", metric, ":", best_t)
        plt.plot(l_threshold,l_score)
        plt.title ( " Score evolution for aglomerative clustering with threshold parameters with metrics " + metric)
        plt.show ()

        ret = best_t
    
    f0 = [element[0] for element in datanp]
    f1 = [element[1] for element in datanp]
    plt.scatter ( f0 , f1 , c = best_model.labels_ , s = 8 )
    title = "Data with aglomerative clustering, " + str(best_model.n_clusters_) + " clusters, linkage : " + str(linkage) + ", metrics : " + str(metric)
    plt.title(title)
    plt.show ()
    print ("score = ", best_score)
    return best_model, ret, best_score



if __name__ == "__main__":
    path = './artificial/'
    databrut = arff.loadarff ( open ( path + "threenorm.arff" , 'r') )
    datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in databrut [ 0 ] ]
    f0 = [element[0] for element in datanp]
    f1 = [element[1] for element in datanp]
    plt.scatter( f0 , f1 , s = 8 )
    plt.title( "Initial Data" )
    plt.show()
    computeAglomerative(datanp, "clusters", 2, 30, 0, 0, 0, 'single', "Calinski-Harabasz")
    computeAglomerative(datanp, "clusters", 2, 30, 0, 0, 0, 'complete', "Davies-Bouldin")
    computeAglomerative(datanp, "clusters", 2, 30, 0, 0, 0, 'average', "Calinski-Harabasz")
    computeAglomerative(datanp, "clusters", 2, 30, 0, 0, 0, 'ward', "Calinski-Harabasz")