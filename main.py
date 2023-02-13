import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import helper as help

from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, SpectralClustering

X, Y = make_moons(n_samples=10, noise=0)

moonsGraph = help.makeMoonsGraph(X)
L = nx.laplacian_matrix(moonsGraph)
A = nx.adjacency_matrix(moonsGraph)
Lnorm = nx.normalized_laplacian_matrix(moonsGraph)
#matrix.A returns matrix view

#print(L.toarray())
eigvalues, eigVectors = np.linalg.eig(Lnorm.A)

#help.findBestEigen(eigV, Y, X)
#print("Largest eigenvalue:", max(e))
#print("Smallest eigenvalue:", min(e))

new_x = []
new_y = []

for r in X:
    new_x.append(r[0])
    new_y.append(r[1])

numbers = {
    "a" : [],
    "b" : [],
    "d" : [],
    "D" : [],
} 

c = 1
euler = 2
dt = 0.1
iterations = 500

def u0(second_eigen, x):
    
    mean = 0
    for i in second_eigen:
        mean += i
    mean /= len(second_eigen)

    val = second_eigen[x] - mean
    if val <= 0:
        return -1
    else:
        return 1

print(eigvalues)
#init a
#for i,vector in enumerate(eigVectors): 




#initialize a



#kmeans = KMeans(n_clusters=2)
#new_data = eigV[0].reshape(-1,1)

#spec = SpectralClustering(n_clusters=4, affinity='nearest_neighbors')

#kmeans.fit(new_data)
#spec.fit(X)
#plt.scatter(new_x, new_y, c=spec.labels_)
#plt.savefig("moons.png")

#pos=nx.get_node_attributes(moonsGraph,'pos')
#ax = plt.subplot(121)
#nx.draw(moonsGraph, pos)
#plt.savefig('graph.png')
