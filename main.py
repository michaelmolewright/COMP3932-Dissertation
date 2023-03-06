import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import helper as util
import time

matplotlib.use('Agg')

from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, SpectralClustering
n = 2000

start = time.time()
X, Y = make_moons(n_samples=n, noise=0.1)
end = time.time()
print(end - start," -- 1. Moons DataSet")

X = np.ndarray.tolist(X)
Y = np.ndarray.tolist(Y)

start = time.time()
newArr = []
for i,x in enumerate(X):
    newArr.append([x,Y[i]])
newArr.sort(key=util.sortingFunction2)

X = []
Y = []
for i in newArr:
    X.append(i[0])
    Y.append(i[1])

end = time.time()
print(end - start," -- 2. Sorting X and Y")


moonsGraph = util.makeMoonsGraph(X)
start = time.time()
moonsGraph = util.n_nearest_Neighbours(moonsGraph,10)
end = time.time()
print(end - start," -- 3. Similarity Function")
#moonsGraph = util.gaussian_weights(moonsGraph, 2)
start = time.time()
Lnorm = nx.normalized_laplacian_matrix(moonsGraph)
end = time.time()
print(end - start," -- 4. Laplacian")
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
start = time.time()
eigens = util.sortEigens(eigvalues, eigVectors)
#for i, val in enumerate(eigens["vectors"]):
  #  print(i, ": ", eigens["values"][i], val)
end = time.time()
print(end - start," -- 5. Eigen Sorting")

c = 1
epsilon = 2
dt = 0.1
iterations = 100

#a = util.a_init(X, eigens["vectors"])
#b = util.b_init(X, eigens["vectors"])
#d = util.d_init(eigens["vectors"])
#D = util.D_init(dt, eigens["values"], c, epsilon)

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

#seg = util.second_eigenvector_segmentation(eigens["vectors"][1])
nodes = list(moonsGraph.nodes)
start = time.time()
seg = util.ginzburg_landau_segmentation(nodes, eigens["values"], eigens["vectors"], 0.1, 1, 2, 500)
end = time.time()
print(end - start," -- 6. Segmentation")
plt.scatter(new_x, new_y, c=seg)
plt.savefig("moons.png")


'''
a = 0
highestVal = 0
highestEig = 0
for j in eigens["vectors"]:
    kmeans.fit(j.reshape(-1, 1))
    #spec.fit(X)
    plt.scatter(new_x, new_y, c=kmeans.labels_)
    plt.savefig("moons.png")
    total = 0
    for i, x in enumerate(Y):
        if kmeans.labels_[i] == x:
            total += 1
    acc = total / len(Y)
    if acc > highestVal:
        highestEig = a
        highestVal = acc
    print("eigen", a, "accuracy = ", acc )
    a += 1

kmeans.fit(eigens["vectors"][highestEig].reshape(-1, 1))
plt.scatter(new_x, new_y, c=kmeans.labels_)
plt.savefig("moons.png")

pos=nx.get_node_attributes(moonsGraph,'pos')

ax = plt.subplot(121)
nx.draw(moonsGraph, pos)
plt.savefig('graph.png')
'''