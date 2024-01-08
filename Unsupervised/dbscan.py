import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
import time
from sklearn import cluster, metrics
from sklearn.cluster import DBSCAN

# Parser un fichier de donnees au format arff
# data est un tableau d ’ exemples avec pour chacun
# la liste des valeurs des features
#
# Dans les jeux de donnees consideres :
# il y a 2 features ( dimension 2 )
# Ex : [[ - 0 . 499261 , -0 . 0612356 ] ,
# [ - 1 . 51369 , 0 . 265446 ] ,
# [ - 1 . 60321 , 0 . 362039 ] , .....
# ]
#
# Note : chaque exemple du jeu de donnees contient aussi un
# numero de cluster . On retire cette information
path = './artificial/'
databrut = arff.loadarff ( open ( path + "2d-4c.arff" , 'r') )
datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in databrut [ 0 ] ]
# Affichage en 2D
# Extraire chaque valeur de features pour en faire une liste
# Ex pour f0 = [ - 0 . 499261 , -1 . 51369 , -1 . 60321 , ...]
# Ex pour f1 = [ - 0 . 0612356 , 0 . 265446 , 0 . 362039 , ...]
f0 = [element[0] for element in datanp]
f1 = [element[1] for element in datanp] # tous les elements de la deuxieme colonne
plt.scatter( f0 , f1 , s = 8 )
plt.title( " Donnees initiales " )
plt.show()


tps1 = time.time()
db = DBSCAN(eps=5, min_samples=10).fit(datanp)
tps2 = time.time()
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

plt.scatter ( f0 , f1 , c = labels , s = 8 )
plt.title ( " Donnees apres clustering DBSCAN"  )
plt.show ()
print ( " nb clusters = " ,n_clusters_, " nb noise point = ", n_noise_, "runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )

