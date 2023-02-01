import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import helper as help

from sklearn.datasets import make_moons
from sklearn.cluster import KMeans

X, Y = make_moons(n_samples=100, noise=0)

moonsGraph = help.makeMoonsGraph(X)
L = nx.laplacian_matrix(moonsGraph)
A = nx.adjacency_matrix(moonsGraph)
Lnorm = nx.normalized_laplacian_matrix(moonsGraph)
#matrix.A returns matrix view

print(L.toarray())
eigW, eigV = np.linalg.eig(L.A)

#help.findBestEigen(eigV, Y, X)
#print("Largest eigenvalue:", max(e))
#print("Smallest eigenvalue:", min(e))

print(eigW)
#print(eigV[0])

new_x = []
new_y = []

for r in X:
    new_x.append(r[0])
    new_y.append(r[1])

kmeans = KMeans(n_clusters=2)
new_data = eigV[1].reshape(-1,1)

kmeans.fit(new_data)
plt.scatter(new_x, new_y, c=kmeans.labels_)
plt.savefig("moons.png")

#pos=nx.get_node_attributes(moonsGraph,'pos')
#ax = plt.subplot(121)
#nx.draw(moonsGraph, pos)
#plt.savefig('graph.png')
