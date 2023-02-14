import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import helper as help

from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, SpectralClustering
n =20
X, Y = make_moons(n_samples=n, noise=0)

moonsGraph = help.makeMoonsGraph(X)
L = nx.laplacian_matrix(moonsGraph)
A = nx.adjacency_matrix(moonsGraph)
Lnorm = nx.normalized_laplacian_matrix(moonsGraph)
#matrix.A returns matrix view

#print(L.toarray())

#Get Eigen Values and vectors
eigvalues, eigVectors = np.linalg.eig(Lnorm.A)

#Sort eigen vectors and values
eigens = []
for i, val in enumerate(eigvalues):
    eigens.append([val, eigVectors[i], i])

def sortingFunc(e):
  return e[0]

eigens.sort(reverse=False , key=sortingFunc)
eigens[0][0] = 0
vec1 = []

for i in range(n):
    vec1.append(1)

eigens[0][1] = vec1



numbers = {
    "a" : [],
    "b" : [],
    "d" : [],
    "D" : [],
} 

c = 1
euler = 2
dt = 0.1
iterations = 100

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


#init a


for i,eig in enumerate(eigens):
    numbers["d"].append(0)
    numbers["D"].append(1 + (dt * ((euler * eig[0]) + c)))
    a = 0
    b=0
    for x in range(n):
        u = u0(eigens[1][1], x)
        a += u * eig[1][x]
        b += u**3 * eig[1][x]

    numbers["a"].append(a)
    numbers["b"].append(b)


finalU = []

for i in range(0,500):
    for j in range(0,len(numbers["a"])):
        numbers["a"][j] = ( (1+ dt/euler + c*dt) * numbers["a"][j] - (dt/euler)*numbers["b"][j] - dt*numbers["d"][j])/numbers["D"][j]
    
    #working out u for each iteration and for all values of x
    u = []
    for f in range(n):
        val = 0
        for k, eig in enumerate(eigens):
            val += numbers["a"][k] * eig[1][f]
        u.append(val)
    for k in range(0,len(numbers["b"])):
        b = 0
        d = 0
        for x in range(n):
            b += u[x] * eigens[k][1][x]
            d += (u[x] - u0(eigens[1][1], x)) * eigens[k][1][x]
        numbers["b"][k] = 0
        numbers["d"][k] = 0
    finalU = u

            
print(finalU)
#initialize a

new_x = []
new_y = []

for r in X:
    new_x.append(r[0])
    new_y.append(r[1])


#kmeans = KMeans(n_clusters=2)
#new_data = eigV[0].reshape(-1,1)

#spec = SpectralClustering(n_clusters=4, affinity='nearest_neighbors')

#kmeans.fit(new_data)
#spec.fit(X)
plt.scatter(new_x, new_y, c=finalU)
plt.savefig("moons.png")

#pos=nx.get_node_attributes(moonsGraph,'pos')
#ax = plt.subplot(121)
#nx.draw(moonsGraph, pos)
#plt.savefig('graph.png')
