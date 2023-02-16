import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import helper as util

matplotlib.use('Agg')

from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, SpectralClustering
n =300
X, Y = make_moons(n_samples=n, noise=0.05)

moonsGraph = util.makeMoonsGraph(X)

Lnorm = nx.normalized_laplacian_matrix(moonsGraph)
#matrix.A returns matrix view

#print(L.toarray())

#Get Eigen Values and vectors
eigvalues, eigVectors = np.linalg.eig(Lnorm.A)

'''
Eigens Dictionary structure 
Values = []
Vectors = []
Key = []
'''
eigens = util.sortEigens(eigvalues, eigVectors)

numbers = {
    "a" : [],
    "b" : [],
    "d" : [],
    "D" : [],
} 

c = 1
epsilon = 2
dt = 0.1
iterations = 100

a = util.a_init(eigens["key"], eigens["vectors"])
b = util.b_init(eigens["key"], eigens["vectors"])
d = util.d_init(eigens["vectors"])
D = util.D_init(dt, eigens["values"], c, epsilon)

#print(a)
#print(b)
#print(d)
#print(D)

#for i in range(10):
 #   a = util.a_nth(a,b,d,D,dt,epsilon,c)
  #  b = util.b_nth(eigens["key"], eigens["vectors"], a )
   # d = util.d_nth(eigens["key"], eigens["vectors"], a )
    #print(a)
    #print(b)
    #print(d)
    #input()

output = []
'''
for i in eigens["key"]:
    print( i, util.segment(util.u_nth(a, eigens["vectors"], i)))
    output.append(util.segment(util.u_nth(a, eigens["vectors"], i)))

'''
for i in range(n):
    output.append(util.u_initial(eigens["vectors"][1], i))
#init a
'''
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

'''

'''
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
'''
'''            
for i in eigVectors:
    for j in eigVectors:
        print(round(np.dot(i, j), 2))
#initialize a



new_x = []
new_y = []

for r in X:
    new_x.append(r[0])
    new_y.append(r[1])
'''
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
plt.scatter(new_x, new_y, c=output)
plt.savefig("moons.png")

#pos=nx.get_node_attributes(moonsGraph,'pos')
#ax = plt.subplot(121)
#nx.draw(moonsGraph, pos)
#plt.savefig('graph.png')
