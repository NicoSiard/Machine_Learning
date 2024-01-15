import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn import cluster, metrics
from sklearn.cluster import DBSCAN
import sys




def computeDBSCAN (datanp, nmin_min, nmin_max, epsilon_min, epsilon_step, epsilon_max, metric):
    """
    datanp : (n_samples, n_features)
    
    search the best epsilon and n_min for clustering with DBSCAN

    metric = "silhouette" or "Davies-Bouldin" or "Calinski-Harabasz"
    
    return best_model, best_nmin, best_epsilon, best_score
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
        
    best_n = -1
    best_e = -1
    best_model = 0
    
    
    for n_min in range(nmin_min,nmin_max+1):
        epsilon = epsilon_min
        while (epsilon <= epsilon_max):
             model = DBSCAN(eps=epsilon, min_samples=n_min)
             model = model.fit(datanp)
             labels = model.labels_
             
             try:
                 #compute metrics and compare with previous iteration
                 if (metric == "silhouette"):
                     score = metrics.silhouette_score(datanp, labels)
                     if (score > best_score):
                         best_score = score
                         best_e = epsilon
                         best_n = n_min
                         best_model = model
                 elif (metric == "Davies-Bouldin"):
                     score = metrics.davies_bouldin_score(datanp, labels)
                     if (score < best_score):
                         best_score = score
                         best_e = epsilon
                         best_n = n_min
                         best_model = model
                 elif (metric == "Calinski-Harabasz"):
                     score = metrics.calinski_harabasz_score(datanp, labels)
                     if (score > best_score):
                         best_score = score
                         best_e = epsilon
                         best_n = n_min
                         best_model = model
                 
             
             except ValueError:
                 print(epsilon, "for epsilon and", n_min, "n_min result in only on label")
                 
             epsilon += epsilon_step
         
        
    
    f0 = [element[0] for element in datanp]
    f1 = [element[1] for element in datanp]
    plt.scatter ( f0 , f1 , c = best_model.labels_ , s = 8 )
    n_clusters_ = len(set(best_model.labels_)) - (1 if -1 in best_model.labels_ else 0)
    n_noise_ = list(best_model.labels_).count(-1)
    plt.title ( " Data after clustering DBSCAN with " + str(n_clusters_) + " clusters and " + str(n_noise_) + " noise points with metrics " + metric)
    plt.show ()
    print ("DBSCAN best epsilon is", best_e, "and best n_min is", best_n)
    print ("score =", best_score)
    return best_model, best_n, best_e, best_score




if __name__ == "__main__":
    path = './artificial/'
    databrut = arff.loadarff ( open ( path + "xclara.arff" , 'r') )
    datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in databrut [ 0 ] ]
    f0 = [element[0] for element in datanp]
    f1 = [element[1] for element in datanp]
    plt.scatter( f0 , f1 , s = 8 )
    plt.title( " Initial Datas" )
    plt.show()
    computeDBSCAN(datanp, 1, 10, 0.1, 0.1, 5, "silhouette")