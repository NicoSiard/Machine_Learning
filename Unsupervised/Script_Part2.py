#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:58:20 2024

@author: raphael tessier and nicolas siard
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn import cluster, metrics
import sys
from k_Means import computeKMeans
from hierarchical import computeAglomerative










if __name__ == "__main__":
    path = './dataset-rapport/'

    name = ['zz2.txt']

    for file in name :
        print (file, " :")
        data = np.loadtxt(path + file)
        f0 = [element[0] for element in data]
        f1 = [element[1] for element in data]
        plt.scatter( f0 , f1 , s = 8 )
        plt.title( file + " Initial Datas" )
        plt.show()
        
        print("   Kmeans :")
        computeKMeans(data, 2, 30, "silhouette")
        print("   Aglomerative :") 
        computeAglomerative(data, "clusters", 2, 30, 0, 0, 0, 'complete', "Calinski-Harabasz")
        computeAglomerative(data, "clusters", 2, 30, 0, 0, 0, 'average', "Calinski-Harabasz")
        computeAglomerative(data, "clusters", 2, 30, 0, 0, 0, 'single', "Calinski-Harabasz")

        
        
        
        
        

